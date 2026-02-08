#!/usr/bin/env julia
# Test LayerNorm gradient correctness specifically for pair-sized tensors.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina, CUDA, Flux, Flux.Zygote, Printf

LaProteina.enable_tf32!()

function maxabsdiff(a, b)
    return maximum(abs.(Array(a) .- Array(b)))
end

println("="^70)
println("LayerNorm Pair Gradient Test")
println("="^70)

# Test with pair feature dimensions
pair_dim = 256
L = 32  # Smaller L for CPU comparison feasibility
B = 2

ln_cpu = PyTorchLayerNorm(pair_dim)
ln_gpu = deepcopy(ln_cpu) |> gpu

# Non-affine version (used in AdaLN)
ln_na_cpu = PyTorchLayerNorm(pair_dim; affine=false)
ln_na_gpu = deepcopy(ln_na_cpu) |> gpu

println("\n1. Affine LN on [$(pair_dim), $(L), $(L), $(B)] pair tensor")
x_cpu = randn(Float32, pair_dim, L, L, B)
x_gpu = CUDA.CuArray(x_cpu)

# Forward
y_cpu = ln_cpu(x_cpu)
y_gpu = ln_gpu(x_gpu)
fwd_diff = maxabsdiff(y_cpu, y_gpu)
@printf("  Forward max diff:  %.8f %s\n", fwd_diff, fwd_diff < 1e-4 ? "[PASS]" : "[FAIL]")

# Backward
g_cpu = Zygote.gradient(x -> sum(ln_cpu(x)), x_cpu)
g_gpu = Zygote.gradient(x -> sum(ln_gpu(x)), x_gpu)
bwd_diff = maxabsdiff(g_cpu[1], g_gpu[1])
@printf("  Backward max diff: %.8f %s\n", bwd_diff, bwd_diff < 1e-3 ? "[PASS]" : "[FAIL]")

println("\n2. Non-affine LN on [$(pair_dim), $(L), $(L), $(B)] pair tensor")
g_na_cpu = Zygote.gradient(x -> sum(ln_na_cpu(x)), x_cpu)
g_na_gpu = Zygote.gradient(x -> sum(ln_na_gpu(x)), x_gpu)
bwd_na_diff = maxabsdiff(g_na_cpu[1], g_na_gpu[1])
@printf("  Backward max diff: %.8f %s\n", bwd_na_diff, bwd_na_diff < 1e-3 ? "[PASS]" : "[FAIL]")

println("\n3. Gradient w.r.t. scale and bias (affine LN)")
# Test parameter gradients
ps_cpu = Flux.params(ln_cpu)
ps_gpu = Flux.params(ln_gpu)

g_ps_cpu = Zygote.gradient(() -> sum(ln_cpu(x_cpu)), ps_cpu)
g_ps_gpu = Zygote.gradient(() -> sum(ln_gpu(x_gpu)), ps_gpu)

scale_diff = maxabsdiff(g_ps_cpu[ln_cpu.scale], g_ps_gpu[ln_gpu.scale])
bias_diff = maxabsdiff(g_ps_cpu[ln_cpu.bias], g_ps_gpu[ln_gpu.bias])
@printf("  Scale grad max diff: %.8f %s\n", scale_diff, scale_diff < 1e-2 ? "[PASS]" : "[FAIL]")
@printf("  Bias grad max diff:  %.8f %s\n", bias_diff, bias_diff < 1e-2 ? "[PASS]" : "[FAIL]")

println("\n4. Timing comparison (pair-sized tensor, L=$L)")
function bench(name, f, n_warmup=3, n_iter=20)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:n_iter; f(); end
    @printf("  %-40s %.3f ms\n", name, t / n_iter * 1000)
end

bench("LN forward (pair)", () -> ln_gpu(x_gpu))
bench("LN backward (pair)", () -> begin
    Zygote.gradient(x -> sum(ln_gpu(x)), x_gpu)
    nothing
end)

# Also test with full L=128
println("\n5. Timing with L=128 (full training size)")
L2 = 128
x_big = CUDA.randn(Float32, pair_dim, L2, L2, B)
ln_big = PyTorchLayerNorm(pair_dim) |> gpu
ln_na_big = PyTorchLayerNorm(pair_dim; affine=false) |> gpu

bench("LN forward (pair, L=128, affine)", () -> ln_big(x_big))
bench("LN forward (pair, L=128, non-affine)", () -> ln_na_big(x_big))
bench("LN backward (pair, L=128, affine)", () -> begin
    Zygote.gradient(x -> sum(ln_big(x)), x_big)
    nothing
end)
bench("LN backward (pair, L=128, non-affine)", () -> begin
    Zygote.gradient(x -> sum(ln_na_big(x)), x_big)
    nothing
end)

println("\n" * "="^70)
println("Test complete.")
println("="^70)
