#!/usr/bin/env julia
# Profile SwiGLU transition with view optimization
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Flux
using Flux: Zygote
using Printf

LaProteina.enable_tf32!()

L = 128; B = 4
token_dim = 768; dim_cond = 256

transition = TransitionADALN(token_dim, dim_cond) |> gpu
x = CUDA.randn(Float32, token_dim, L, B)
cond = CUDA.randn(Float32, dim_cond, L, B)
mask = CUDA.ones(Float32, L, B)

function bench(name, f, n_warmup=5, n_iter=20)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:n_iter; f(); end
    @printf("  %-55s %.2f ms\n", name, t / n_iter * 1000)
end

println("="^70)
println("SwiGLU Transition After View Optimization")
println("="^70)

tr = transition.transition

println("\n--- Forward ---")
bench("TransitionADALN forward", () -> transition(x, cond, mask))
bench("SwiGLUTransition forward", () -> tr(x, mask))
bench("SwiGLU forward", () -> begin
    xe = tr.linear_in(x)
    tr.swiglu(xe)
end)

println("\n--- Backward ---")
bench("TransitionADALN backward", () -> begin
    Zygote.gradient(x -> sum(transition(x, cond, mask)), x)
end)
bench("SwiGLUTransition backward", () -> begin
    Zygote.gradient(x -> sum(tr(x, mask)), x)
end)

# SwiGLU backward with views
x_expanded = tr.linear_in(x)
bench("SwiGLU backward (view)", () -> begin
    Zygote.gradient(xe -> sum(tr.swiglu(xe)), x_expanded)
end)

# Dense layers backward
bench("Dense 768→6144 backward (w.r.t. both)", () -> begin
    Zygote.gradient(m -> sum(m(x)), tr.linear_in)
end)
bench("Dense 3072→768 backward (w.r.t. both)", () -> begin
    xg = tr.swiglu(x_expanded)
    Zygote.gradient(m -> sum(m(xg)), tr.linear_out)
end)

# AdaLN components
bench("AdaLN backward", () -> begin
    Zygote.gradient(x -> sum(transition.adaln(x, cond, mask)), x)
end)
bench("AdaptiveOutputScale backward", () -> begin
    Zygote.gradient(x -> sum(transition.scale_output(x, cond, mask)), x)
end)

# The actual mask multiplication
println("\n--- Mask operations ---")
mask_exp = reshape(mask, 1, L, B)
bench("x .* mask_exp", () -> x .* mask_exp)
bench("(1 .- mask_exp) broadcast", () -> (1.0f0 .- mask_exp))

println("\n" * "="^70)
println("Done.")
println("="^70)
