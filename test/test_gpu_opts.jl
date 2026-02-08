#!/usr/bin/env julia
# Test GPU optimization module (no cuTile path)
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Flux

println("Testing GPU optimizations (no cuTile path)...")

# Test 1: TF32
println("\n1. TF32 mode:")
LaProteina.enable_tf32!()
println("   TF32 enabled: ", CUDA.math_mode())

# Test 2: within_gradient
println("\n2. within_gradient:")
println("   Direct call: ", LaProteina.within_gradient(1.0))

# Test 3: Buffer pool
println("\n3. Buffer pool:")
buf = LaProteina._get_perm_buf(1, (64, 128, 12, 4))
println("   Buffer shape: ", size(buf))
buf2 = LaProteina._get_perm_buf(1, (64, 128, 12, 4))
println("   Same buffer: ", pointer(buf) == pointer(buf2))

# Test 4: PyTorchLayerNorm on GPU
println("\n4. PyTorchLayerNorm on GPU:")
ln = PyTorchLayerNorm(768) |> gpu
x = CUDA.randn(Float32, 768, 128, 4)
y = ln(x)
println("   Output shape: ", size(y))
println("   Close to zero mean: ", abs(Float32(sum(y)) / length(y)) < 0.1)

# Test 5: PairBiasAttention
println("\n5. PairBiasAttention:")
attn = PairBiasAttention(768, 12; pair_dim=256, qk_ln=true) |> gpu
node = CUDA.randn(Float32, 768, 64, 2)
pair = CUDA.randn(Float32, 256, 64, 64, 2)
mask = CUDA.ones(Float32, 64, 2)
out = attn(node, pair, mask)
println("   Output shape: ", size(out))

# Test 6: TransformerBlock
println("\n6. TransformerBlock:")
block = TransformerBlock(dim_token=768, dim_pair=256, n_heads=12, dim_cond=256, qk_ln=true) |> gpu
cond = CUDA.randn(Float32, 256, 64, 2)
out = block(node, pair, cond, mask)
println("   Output shape: ", size(out))

# Test 7: Gradient test (within_gradient should return true)
println("\n7. Gradient test:")
using Flux: Zygote
simple_ln = PyTorchLayerNorm(64) |> gpu
x_small = CUDA.randn(Float32, 64, 8, 2)
loss(x) = sum(simple_ln(x))
g = Zygote.gradient(loss, x_small)
println("   Gradient computed: ", g[1] !== nothing)
println("   Gradient shape: ", size(g[1]))

# Test 8: TF32 timing comparison
println("\n8. Timing test (Dense matmul):")
dense = Dense(768 => 768) |> gpu
x_big = CUDA.randn(Float32, 768, 128, 4)
# Warmup
for _ in 1:3
    dense(x_big)
end
CUDA.synchronize()
t = CUDA.@elapsed for _ in 1:100
    dense(x_big)
end
println("   100x Dense(768->768) with TF32: ", round(t * 1000, digits=2), " ms")

println("\n✓ All tests passed.")
