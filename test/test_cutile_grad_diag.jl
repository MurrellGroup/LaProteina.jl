#!/usr/bin/env julia
# Diagnostic: Run a few training steps with cuTile and dump gradient stats.
# Compare with nocutile by running with LAPROTEINA_NOCUTILE=1.
# Usage: julia -t 8 test/test_cutile_grad_diag.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CUDA
using Flux
using Zygote
using Statistics
using Printf
using Random

using LaProteina

println("cuTile available: ", LaProteina._HAS_CUTILE)
println("No overrides: ", LaProteina._NO_OVERRIDES)
LaProteina.enable_tf32!()

# ============================================================================
# Load model and data
# ============================================================================

println("\n=== Loading Model ===")
model_path = joinpath(@__DIR__, "..", "models", "branching_score_network_bf.bson")
if !isfile(model_path)
    model_path = joinpath(@__DIR__, "..", "models", "branching_score_network.bson")
end
model = LaProteina.load_branching_model(model_path) |> gpu
println("Model loaded")

println("\n=== Loading Data ===")
data_dir = joinpath(@__DIR__, "..", "data", "shards")
proteins = LaProteina.load_training_data(data_dir; max_shards=1)
println("Loaded $(length(proteins)) proteins")

# Create DataLoader
loader = LaProteina.DataLoader(proteins, 4; max_length=150, parallel=true)

# ============================================================================
# Training step function (same as train_branching_full.jl)
# ============================================================================

function compute_branching_loss(model, batch, mask)
    raw = LaProteina.prepare_training_batch(model, batch, mask)
    out = LaProteina.forward_branching_from_raw_features_gpu(model, raw.features)

    ca_pred = out[:bb_ca][model.base.output_param]
    ll_pred = out[:local_latents][model.base.output_param]
    ca_tgt = raw.targets[:bb_ca]
    ll_tgt = raw.targets[:local_latents]

    mask_exp = reshape(mask, 1, size(mask)...)

    ca_diff = (ca_pred .- ca_tgt) .* mask_exp
    ll_diff = (ll_pred .- ll_tgt) .* mask_exp

    n_valid = max(sum(mask), 1f0)

    ca_loss_raw = sum(ca_diff .^ 2) / n_valid
    ll_loss_raw = sum(ll_diff .^ 2) / n_valid

    ca_loss = LaProteina.softclamp(ca_loss_raw * 2f0)
    ll_loss = LaProteina.softclamp(ll_loss_raw * 0.1f0)

    return ca_loss + ll_loss
end

# ============================================================================
# Run diagnostic
# ============================================================================

println("\n=== Running Gradient Diagnostics ===")
println("Running 5 optimizer steps...")

Random.seed!(42)

# Use same optimizer as training
opt_state = Flux.setup(Muon(eta=Float64(1e-5)), model)

for step in 1:5
    batch_data = iterate(loader)
    if isnothing(batch_data)
        loader = LaProteina.DataLoader(proteins, 4; max_length=150, parallel=true)
        batch_data = iterate(loader)
    end
    batch, _ = batch_data

    batch_gpu = LaProteina.batch_to_gpu(batch)
    mask = batch_gpu.mask

    # Forward + backward
    loss_val, grads = Flux.withgradient(model) do m
        compute_branching_loss(m, batch_gpu, mask)
    end

    loss_f = Float64(loss_val)
    loss_nan = isnan(loss_f)

    @printf("Step %d: loss=%.4f%s\n", step, loss_f, loss_nan ? " [NaN!]" : "")

    # Analyze gradients
    grad_tree = grads[1]
    n_nan = 0
    n_inf = 0
    n_total = 0
    max_grad = 0f0
    max_grad_name = ""

    function walk_grads(tree, prefix="")
        if tree isa NamedTuple || tree isa Tuple
            for (i, name) in enumerate(tree isa NamedTuple ? keys(tree) : 1:length(tree))
                walk_grads(tree[i isa Int ? i : name], "$(prefix).$(name)")
            end
        elseif tree isa AbstractArray
            arr = Array(tree)
            n_nan_local = count(isnan, arr)
            n_inf_local = count(isinf, arr)
            max_local = maximum(abs.(arr[.!isnan.(arr) .& .!isinf.(arr)]))
            if n_nan_local > 0 || n_inf_local > 0 || max_local > 100
                @printf("  GRAD %s: max=%.3e, nNaN=%d, nInf=%d, shape=%s\n",
                        prefix, max_local, n_nan_local, n_inf_local, string(size(tree)))
            end
            if max_local > max_grad && !isnan(max_local)
                max_grad = max_local
                max_grad_name = prefix
            end
        elseif !isnothing(tree)
            # Skip nothing entries
        end
    end

    try
        walk_grads(grad_tree)
    catch e
        println("  Error walking gradients: ", e)
    end

    @printf("  Max grad: %.4e at %s\n", max_grad, max_grad_name)

    if loss_nan
        println("  STOPPING — loss is NaN")
        break
    end

    # Optimizer step
    Flux.update!(opt_state, model, grads[1])

    # Check model weights for NaN
    n_param_nan = 0
    for p in Flux.params(model)
        arr = Array(p)
        n_param_nan += count(isnan, arr)
    end
    if n_param_nan > 0
        @printf("  WARNING: %d NaN values in model weights after step %d\n", n_param_nan, step)
        break
    end
end

println("\n=== Done ===")
