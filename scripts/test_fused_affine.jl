#!/usr/bin/env julia
# Test that fused affine + to_bias produces same results as standard path.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Flux
using Flux: Zygote
using Printf

LaProteina.enable_tf32!()

L = 32; B = 2; pair_dim = 256; n_heads = 12

# Create PairBiasAttention-like components
pair_norm = PyTorchLayerNorm(pair_dim) |> gpu
to_bias = Dense(pair_dim => n_heads; bias=false) |> gpu

pair = CUDA.randn(Float32, pair_dim, L, L, B)
pair_eps = pair_norm.ϵ
pair_normed = LaProteina.pytorch_normalise(pair; dims=1, eps=pair_eps)

# Standard path: affine then to_bias
standard = to_bias(pair_norm(pair))

# Fused path: _apply_pair_affine
fused = LaProteina._apply_pair_affine(pair_normed, pair_norm, to_bias)

maxdiff = maximum(abs.(Array(standard) .- Array(fused)))
@printf("Forward max diff: %.8f %s\n", maxdiff, maxdiff < 1e-4 ? "[PASS]" : "[FAIL]")

# Backward test: gradient w.r.t pair_normed
g_standard = Zygote.gradient(p -> begin
    pn = LaProteina.pytorch_normalise(p; dims=1, eps=pair_eps)
    sum(to_bias(pair_norm.scale .* pn .+ pair_norm.bias))
end, pair)

g_fused = Zygote.gradient(p -> begin
    pn = LaProteina.pytorch_normalise(p; dims=1, eps=pair_eps)
    sum(LaProteina._apply_pair_affine(pn, pair_norm, to_bias))
end, pair)

grad_diff = maximum(abs.(Array(g_standard[1]) .- Array(g_fused[1])))
@printf("Backward (pair) max diff: %.8f %s\n", grad_diff, grad_diff < 1e-3 ? "[PASS]" : "[FAIL]")

# Backward test: gradient w.r.t. model params (scale, bias, W)
g_model_std = Zygote.gradient(() -> begin
    pn = LaProteina.pytorch_normalise(pair; dims=1, eps=pair_eps)
    sum(to_bias(pair_norm.scale .* pn .+ pair_norm.bias))
end, Flux.params(pair_norm, to_bias))

g_model_fused = Zygote.gradient(() -> begin
    pn = LaProteina.pytorch_normalise(pair; dims=1, eps=pair_eps)
    sum(LaProteina._apply_pair_affine(pn, pair_norm, to_bias))
end, Flux.params(pair_norm, to_bias))

for p in Flux.params(pair_norm, to_bias)
    if !isnothing(g_model_std[p]) && !isnothing(g_model_fused[p])
        d = maximum(abs.(Array(g_model_std[p]) .- Array(g_model_fused[p])))
        @printf("Param grad diff (size %s): %.8f %s\n", size(p), d, d < 1e-3 ? "[PASS]" : "[FAIL]")
    end
end

# Performance comparison at full size
println("\n--- Full size timing (L=128, B=4) ---")
L = 128; B = 4
pair_full = CUDA.randn(Float32, pair_dim, L, L, B)
pair_normed_full = LaProteina.pytorch_normalise(pair_full; dims=1, eps=pair_eps)

function bench(name, f, n_warmup=5, n_iter=20)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:n_iter; f(); end
    @printf("  %-50s %.2f ms\n", name, t / n_iter * 1000)
end

bench("Standard: affine broadcast + to_bias", () -> begin
    out = @. pair_normed_full * pair_norm.scale + pair_norm.bias
    to_bias(out)
end)

bench("Fused: _apply_pair_affine", () -> begin
    LaProteina._apply_pair_affine(pair_normed_full, pair_norm, to_bias)
end)

bench("Standard backward", () -> begin
    Zygote.gradient(pn -> begin
        out = @. pn * pair_norm.scale + pair_norm.bias
        sum(to_bias(out))
    end, pair_normed_full)
end)

bench("Fused backward", () -> begin
    Zygote.gradient(pn -> begin
        sum(LaProteina._apply_pair_affine(pn, pair_norm, to_bias))
    end, pair_normed_full)
end)

println("\nDone!")
