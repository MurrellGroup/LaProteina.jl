#!/usr/bin/env julia
# Profile LayerNorm forward implementations to find regression.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina, CUDA, Flux, Printf, Statistics

LaProteina.enable_tf32!()

pair_dim = 256; L = 128; B = 4

function bench(name, f, n_warmup=5, n_iter=50)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:n_iter; f(); end
    @printf("  %-50s %.3f ms\n", name, t / n_iter * 1000)
end

x = CUDA.randn(Float32, pair_dim, L, L, B)

println("="^70)
println("LayerNorm Forward Profile")
println("="^70)

# Direct pytorch_normalise (what the original code does)
println("\n--- Direct operations ---")
bench("mean(x, dims=1)", () -> mean(x; dims=1))
bench("x .- mean", () -> x .- mean(x; dims=1))

centered = x .- mean(x; dims=1)
bench("centered.^2", () -> centered .^ 2)
bench("mean(centered.^2, dims=1)", () -> mean(centered .^ 2; dims=1))

sigma2 = mean(centered .^ 2; dims=1)
bench("inv_std = 1/sqrt(sigma2+eps)", () -> @. 1 / sqrt(sigma2 + 1f-5))

inv_std = @. 1 / sqrt(sigma2 + 1f-5)
bench("normed = centered .* inv_std", () -> centered .* inv_std)

normed = centered .* inv_std
scale = CUDA.ones(Float32, pair_dim)
bias = CUDA.zeros(Float32, pair_dim)
bench("y = normed .* scale .+ bias", () -> @. normed * scale + bias)

println("\n--- Full implementations ---")

# Method 1: Original pytorch_normalise (Zygote will trace this)
bench("pytorch_normalise (original)", () -> LaProteina.pytorch_normalise(x; dims=1, eps=1f-5))

# Method 2: _pytorch_layernorm_fwd (our custom function)
bench("_pytorch_layernorm_fwd (affine)", () -> LaProteina._pytorch_layernorm_fwd(x, scale, bias, 1f-5, 1:1))
bench("_pytorch_layernorm_fwd (no-affine)", () -> LaProteina._pytorch_layernorm_fwd(x, nothing, nothing, 1f-5, 1:1))

# Method 3: Full PyTorchLayerNorm (dispatches to our override)
ln = PyTorchLayerNorm(pair_dim) |> gpu
bench("PyTorchLayerNorm (full, affine)", () -> ln(x))

ln_na = PyTorchLayerNorm(pair_dim; affine=false) |> gpu
bench("PyTorchLayerNorm (full, non-affine)", () -> ln_na(x))

# Method 4: What var() computes
bench("var(x, dims=1, corrected=false)", () -> var(x; dims=1, corrected=false))

println("\n--- Memory allocation breakdown ---")
println("  Tensor size: $(pair_dim) x $(L) x $(L) x $(B) = $(pair_dim * L * L * B) floats = $(pair_dim * L * L * B * 4 / 1e6) MB")

# Count forward allocations for _pytorch_layernorm_fwd
println("\n  _pytorch_layernorm_fwd creates:")
println("    μ:        [1, $(L), $(L), $(B)]  = $(L*L*B * 4 / 1e6) MB")
println("    centered: [$(pair_dim), $(L), $(L), $(B)] = $(pair_dim*L*L*B * 4 / 1e6) MB")
println("    σ²:       [1, $(L), $(L), $(B)]  = $(L*L*B * 4 / 1e6) MB")
println("    inv_std:  [1, $(L), $(L), $(B)]  = $(L*L*B * 4 / 1e6) MB")
println("    normed:   [$(pair_dim), $(L), $(L), $(B)] = $(pair_dim*L*L*B * 4 / 1e6) MB")
println("    y:        [$(pair_dim), $(L), $(L), $(B)] = $(pair_dim*L*L*B * 4 / 1e6) MB")
total = (3 * pair_dim * L * L * B + 3 * L * L * B) * 4
println("    Total: $(total / 1e6) MB")

println("\n" * "="^70)
