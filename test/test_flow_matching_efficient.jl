# Test flow matching training with efficient forward pass

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using Flux
using CUDA
using Statistics
using Random

Random.seed!(42)

println("=" ^ 60)
println("Flow Matching Training with Efficient Forward")
println("=" ^ 60)

# Check GPU
println("\n=== GPU Status ===")
if CUDA.functional()
    println("CUDA is functional!")
    println("Device: ", CUDA.device())
else
    println("CUDA not functional, running on CPU")
end

# Create ScoreNetwork
println("\n=== Creating ScoreNetwork ===")
latent_dim = 8
model = ScoreNetwork(
    n_layers=14,
    token_dim=768,
    pair_dim=256,
    n_heads=12,
    dim_cond=256,
    latent_dim=latent_dim,
    output_param=:v,
    qk_ln=true,
    update_pair_repr=false
)

# Load pretrained weights
weights_path = joinpath(@__DIR__, "..", "weights", "score_network.npz"))
if isfile(weights_path)
    println("Loading pretrained weights...")
    load_score_network_weights!(model, weights_path)
end

model_gpu = model |> gpu
println("Model on GPU")

# Flow matching loss function using efficient forward
function efficient_flow_loss(model, batch)
    # batch should have x_0 (noise), x_1 (target), t, mask
    x_0_ca = batch[:x_0][:bb_ca]
    x_0_ll = batch[:x_0][:local_latents]
    x_1_ca = batch[:x_1][:bb_ca]
    x_1_ll = batch[:x_1][:local_latents]
    t_ca = batch[:t][:bb_ca]
    t_ll = batch[:t][:local_latents]
    mask = batch[:mask]

    L, B = size(mask)

    # Interpolate: x_t = (1-t) * x_0 + t * x_1
    t_ca_exp = reshape(t_ca, 1, 1, B)
    t_ll_exp = reshape(t_ll, 1, 1, B)

    x_t_ca = (1f0 .- t_ca_exp) .* x_0_ca .+ t_ca_exp .* x_1_ca
    x_t_ll = (1f0 .- t_ll_exp) .* x_0_ll .+ t_ll_exp .* x_1_ll

    # Target velocity: v = x_1 - x_0
    v_target_ca = x_1_ca .- x_0_ca
    v_target_ll = x_1_ll .- x_0_ll

    # Create efficient batch
    eff_batch = EfficientScoreNetworkBatch(x_t_ca, x_t_ll, t_ca, t_ll, mask)

    # Forward pass (with gradients through projection layers)
    output = forward_efficient(model, eff_batch)

    v_pred_ca = output[:bb_ca][:v]
    v_pred_ll = output[:local_latents][:v]

    # MSE loss with masking
    mask_seq = reshape(mask, 1, L, B)

    diff_ca = (v_pred_ca .- v_target_ca).^2 .* mask_seq
    diff_ll = (v_pred_ll .- v_target_ll).^2 .* mask_seq

    n_active = sum(mask)
    loss_ca = sum(diff_ca) / (3f0 * n_active + 1f-8)
    loss_ll = sum(diff_ll) / (Float32(latent_dim) * n_active + 1f-8)

    return loss_ca + loss_ll
end

# Create test data
println("\n=== Creating Training Data ===")
L = 20
B = 2
T = Float32

# X_1: target (simulated encoder output)
x_1 = Dict(
    :bb_ca => CUDA.randn(T, 3, L, B) * 0.3f0,
    :local_latents => CUDA.randn(T, latent_dim, L, B)
)

# X_0: noise
x_0 = Dict(
    :bb_ca => CUDA.randn(T, 3, L, B),
    :local_latents => CUDA.randn(T, latent_dim, L, B)
)

# Time
t = Dict(:bb_ca => CUDA.rand(T, B), :local_latents => CUDA.rand(T, B))

# Mask
mask = CUDA.ones(T, L, B)
mask[end-2:end, :] .= 0f0

batch = Dict{Symbol, Any}(
    :x_0 => x_0,
    :x_1 => x_1,
    :t => t,
    :mask => mask
)

println("Data shapes: L=$L, B=$B")

# Test loss computation
println("\n=== Testing Loss Computation ===")
loss_val = efficient_flow_loss(model_gpu, batch)
println("Initial loss: ", loss_val)

# Test gradient computation
println("\n=== Testing Gradient Computation ===")
_, grads = Flux.withgradient(model_gpu) do m
    efficient_flow_loss(m, batch)
end
grads = grads[1]

# Check gradients
println("Gradient check: gradients computed successfully")
println("  Gradients are non-nothing: $(grads !== nothing)")

# Training loop test
println("\n=== Training Loop Test (5 steps) ===")
opt_state = Flux.setup(Adam(1e-4), model_gpu)

for step in 1:5
    # Generate new random times each step
    t_new = Dict(
        :bb_ca => CUDA.rand(T, B),
        :local_latents => CUDA.rand(T, B)
    )
    batch[:t] = t_new

    loss, grads = Flux.withgradient(model_gpu) do m
        efficient_flow_loss(m, batch)
    end

    Flux.update!(opt_state, model_gpu, grads[1])

    println("  Step $step: loss = $(round(loss, digits=4))")
end

# Timing: efficient training step
println("\n=== Training Step Timing (Efficient Forward) ===")

# Warmup
_, _ = Flux.withgradient(m -> efficient_flow_loss(m, batch), model_gpu)
CUDA.synchronize()

# Time efficient training step
n_iters = 10
t_eff = @elapsed begin
    for _ in 1:n_iters
        loss, grads = Flux.withgradient(m -> efficient_flow_loss(m, batch), model_gpu)
    end
    CUDA.synchronize()
end
println("  Efficient forward: $(round(t_eff/n_iters * 1000, digits=2)) ms/step")
println("  (includes forward + backward pass)")

# Note: Original extract_raw_features doesn't work with gradients due to fill! in bin_values
println("\n  Note: The efficient forward uses Zygote.@ignore for feature computation,")
println("  allowing gradients to flow only through the trainable projection layers.")

println("\n" * "=" ^ 60)
println("Flow Matching Training Test COMPLETED")
println("=" ^ 60)
