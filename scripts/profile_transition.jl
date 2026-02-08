#!/usr/bin/env julia
# Profile SwiGLU transition backward in detail
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Flux
using Flux: Zygote
using Printf

LaProteina.enable_tf32!()

L = 128; B = 4
token_dim = 768; dim_cond = 256

# Create the full TransitionADALN
transition = TransitionADALN(token_dim, dim_cond) |> gpu
x = CUDA.randn(Float32, token_dim, L, B)
cond = CUDA.randn(Float32, dim_cond, L, B)
mask = CUDA.ones(Float32, L, B)

function bench(name, f, n_warmup=3, n_iter=10)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:n_iter; f(); end
    @printf("  %-55s %.2f ms\n", name, t / n_iter * 1000)
end

println("="^70)
println("SwiGLU Transition Profiling (L=$L, B=$B, token=$token_dim)")
println("="^70)

# Full TransitionADALN
println("\n--- Forward ---")
bench("TransitionADALN forward", () -> transition(x, cond, mask))
bench("SwiGLUTransition forward", () -> transition.transition(x, mask))
bench("AdaLN forward", () -> transition.adaln(x, cond, mask))
bench("AdaptiveOutputScale forward", () -> transition.scale_output(x, cond, mask))

# Breakdown of SwiGLUTransition
tr = transition.transition
bench("Dense 768→6144 forward", () -> tr.linear_in(x))
x_expanded = tr.linear_in(x)
bench("SwiGLU forward", () -> tr.swiglu(x_expanded))
x_gated = tr.swiglu(x_expanded)
bench("Dense 3072→768 forward", () -> tr.linear_out(x_gated))

println("\n--- Backward ---")
bench("TransitionADALN backward", () -> begin
    Zygote.gradient(x -> sum(transition(x, cond, mask)), x)
end)

bench("SwiGLUTransition backward", () -> begin
    Zygote.gradient(x -> sum(transition.transition(x, mask)), x)
end)

bench("AdaLN backward", () -> begin
    Zygote.gradient(x -> sum(transition.adaln(x, cond, mask)), x)
end)

bench("AdaptiveOutputScale backward", () -> begin
    Zygote.gradient(x -> sum(transition.scale_output(x, cond, mask)), x)
end)

# SwiGLUTransition sub-component backward
bench("Dense 768→6144 backward", () -> begin
    Zygote.gradient(x -> sum(tr.linear_in(x)), x)
end)

bench("SwiGLU backward", () -> begin
    Zygote.gradient(xe -> sum(tr.swiglu(xe)), x_expanded)
end)

bench("Dense 3072→768 backward", () -> begin
    Zygote.gradient(xg -> sum(tr.linear_out(xg)), x_gated)
end)

# SwiGLU array slicing cost
println("\n--- SwiGLU internals ---")
dim = size(x_expanded, 1) ÷ 2
bench("array slicing (x[1:dim, ...])", () -> begin
    x1 = x_expanded[1:dim, :, :]
    x2 = x_expanded[dim+1:end, :, :]
    nothing
end)

bench("swish broadcast", () -> begin
    x1 = x_expanded[1:dim, :, :]
    x2 = x_expanded[dim+1:end, :, :]
    NNlib.swish.(x2) .* x1
end)

bench("swish broadcast (fused @.)", () -> begin
    x1 = x_expanded[1:dim, :, :]
    x2 = x_expanded[dim+1:end, :, :]
    @. NNlib.swish(x2) * x1
end)

# Mask broadcast cost
println("\n--- Mask broadcast ---")
mask_exp = reshape(mask, 1, L, B)
bench("mask broadcast (x .* mask_exp)", () -> x .* mask_exp)

println("\n" * "="^70)
println("Profiling complete.")
println("="^70)
