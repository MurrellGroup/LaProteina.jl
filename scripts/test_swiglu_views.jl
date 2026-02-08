#!/usr/bin/env julia
# Test SwiGLU with views vs copies
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Flux
using Flux: Zygote
using Printf

LaProteina.enable_tf32!()

L = 128; B = 4; dim_inner = 3072

x = CUDA.randn(Float32, dim_inner * 2, L, B)

function bench(name, f, n_warmup=5, n_iter=20)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:n_iter; f(); end
    @printf("  %-55s %.2f ms\n", name, t / n_iter * 1000)
end

println("="^70)
println("SwiGLU View vs Copy Test")
println("="^70)

# Standard: copy-based
function swiglu_copy(x)
    dim = size(x, 1) ÷ 2
    x1 = x[1:dim, :, :]
    x2 = x[dim+1:end, :, :]
    return NNlib.swish.(x2) .* x1
end

# View-based
function swiglu_view(x)
    dim = size(x, 1) ÷ 2
    x1 = @view x[1:dim, :, :]
    x2 = @view x[dim+1:end, :, :]
    return NNlib.swish.(x2) .* x1
end

# Reshape-based: [2*dim, L, B] -> [dim, 2, L, B] then select
function swiglu_reshape(x)
    dim = size(x, 1) ÷ 2
    x_r = reshape(x, dim, 2, size(x, 2), size(x, 3))
    x1 = x_r[:, 1, :, :]
    x2 = x_r[:, 2, :, :]
    return NNlib.swish.(x2) .* x1
end

# Fused single kernel approach
function swiglu_fused(x)
    dim = size(x, 1) ÷ 2
    x1 = @view x[1:dim, :, :]
    x2 = @view x[dim+1:end, :, :]
    return @. NNlib.swish(x2) * x1
end

println("\n--- Forward ---")
bench("SwiGLU copy", () -> swiglu_copy(x))
bench("SwiGLU view", () -> swiglu_view(x))
bench("SwiGLU reshape", () -> swiglu_reshape(x))
bench("SwiGLU fused (@. with views)", () -> swiglu_fused(x))

println("\n--- Backward ---")
bench("SwiGLU copy backward", () -> begin
    Zygote.gradient(x -> sum(swiglu_copy(x)), x)
end)

bench("SwiGLU view backward", () -> begin
    Zygote.gradient(x -> sum(swiglu_view(x)), x)
end)

bench("SwiGLU reshape backward", () -> begin
    Zygote.gradient(x -> sum(swiglu_reshape(x)), x)
end)

bench("SwiGLU fused backward", () -> begin
    Zygote.gradient(x -> sum(swiglu_fused(x)), x)
end)

println("\n--- Correctness check ---")
y_copy = swiglu_copy(x)
y_view = swiglu_view(x)
y_reshape = swiglu_reshape(x)
y_fused = swiglu_fused(x)

@printf("view vs copy: %.8f\n", maximum(abs.(Array(y_copy) .- Array(y_view))))
@printf("reshape vs copy: %.8f\n", maximum(abs.(Array(y_copy) .- Array(y_reshape))))
@printf("fused vs copy: %.8f\n", maximum(abs.(Array(y_copy) .- Array(y_fused))))

g_copy = Zygote.gradient(x -> sum(swiglu_copy(x)), x)[1]
g_view = Zygote.gradient(x -> sum(swiglu_view(x)), x)[1]
g_reshape = Zygote.gradient(x -> sum(swiglu_reshape(x)), x)[1]
g_fused = Zygote.gradient(x -> sum(swiglu_fused(x)), x)[1]

@printf("view grad vs copy: %.8f\n", maximum(abs.(Array(g_copy) .- Array(g_view))))
@printf("reshape grad vs copy: %.8f\n", maximum(abs.(Array(g_copy) .- Array(g_reshape))))
@printf("fused grad vs copy: %.8f\n", maximum(abs.(Array(g_copy) .- Array(g_fused))))

println("\nDone!")
