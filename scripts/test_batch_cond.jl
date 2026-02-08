#!/usr/bin/env julia
# Test batch-level pair conditioning optimization
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
include(joinpath(@__DIR__, "..", "src", "branching", "branching_score_network.jl"))
using CUDA
using Flux
using Flux: Zygote
using Printf

LaProteina.enable_tf32!()

L = 128; B = 4

base = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256,
    n_heads=12, dim_cond=256, qk_ln=true,
    update_pair_repr=false
) |> gpu

model = BranchingScoreNetwork(base) |> gpu

seq_raw_dim = size(base.init_repr_factory.projection.weight, 2)
cond_raw_dim = size(base.cond_factory.projection.weight, 2)
pair_raw_dim = size(base.pair_rep_builder.init_repr_factory.projection.weight, 2)
pair_cond_raw_dim = size(base.pair_rep_builder.cond_factory.projection.weight, 2)

raw_features = ScoreNetworkRawFeatures(
    CUDA.randn(Float32, seq_raw_dim, L, B),
    CUDA.randn(Float32, cond_raw_dim, L, B),
    CUDA.randn(Float32, pair_raw_dim, L, L, B),
    CUDA.randn(Float32, pair_cond_raw_dim, L, L, B),
    CUDA.ones(Float32, L, B)
)

println("="^70)
println("Batch-Level Pair Conditioning Test")
println("="^70)

# 1. Forward correctness: compare GPU (batch cond) vs standard (full cond)
println("\n--- Forward Correctness ---")
out_std = forward_branching_from_raw_features(model, raw_features)
out_gpu = forward_branching_from_raw_features_gpu(model, raw_features)

for key in [:bb_ca, :local_latents]
    d = maximum(abs.(Array(out_std[key][:v]) .- Array(out_gpu[key][:v])))
    pass = d < 1e-3
    @printf("  %-20s max diff: %.8f %s\n", key, d, pass ? "PASS" : "FAIL")
end
for key in [:split, :del]
    d = maximum(abs.(Array(out_std[key]) .- Array(out_gpu[key])))
    pass = d < 1e-3
    @printf("  %-20s max diff: %.8f %s\n", key, d, pass ? "PASS" : "FAIL")
end

# 2. Backward correctness: compare gradients
println("\n--- Backward Correctness ---")
g_std = Zygote.gradient(m -> begin
    out = forward_branching_from_raw_features(m, raw_features)
    sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v]) + sum(out[:split]) + sum(out[:del])
end, model)

g_gpu = Zygote.gradient(m -> begin
    out = forward_branching_from_raw_features_gpu(m, raw_features)
    sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v]) + sum(out[:split]) + sum(out[:del])
end, model)

max_rel = 0.0
n_checked = 0
for i in 1:14
    bs = g_std[1].base.transformer_layers[i]
    bg = g_gpu[1].base.transformer_layers[i]
    for (name, gs, gg) in [
        ("L$i to_qkv.W", bs.mha.mha.to_qkv.weight, bg.mha.mha.to_qkv.weight),
        ("L$i to_out.W", bs.mha.mha.to_out.weight, bg.mha.mha.to_out.weight),
    ]
        if isnothing(gs) || isnothing(gg); continue; end
        a, b = Array(gs), Array(gg)
        d = maximum(abs.(a .- b))
        m_val = maximum(abs.(a))
        rel = d / (m_val + 1e-8)
        global max_rel = max(max_rel, rel)
        global n_checked += 1
    end
end

# Check pair conditioning gradients specifically
function check_grads_recursive!(gs, gg, max_rel_ref, n_checked_ref)
    if gs isa AbstractArray && gg isa AbstractArray
        a, b = Array(gs), Array(gg)
        d = maximum(abs.(a .- b))
        m_val = maximum(abs.(a))
        rel = d / (m_val + 1e-8)
        max_rel_ref[] = max(max_rel_ref[], rel)
        n_checked_ref[] += 1
    elseif gs isa NamedTuple && gg isa NamedTuple
        for k in keys(gs)
            if haskey(gg, k) && !isnothing(gs[k]) && !isnothing(gg[k])
                check_grads_recursive!(gs[k], gg[k], max_rel_ref, n_checked_ref)
            end
        end
    end
end

if !isnothing(g_std[1].base.pair_rep_builder) && !isnothing(g_gpu[1].base.pair_rep_builder)
    mr_ref = Ref(max_rel)
    nc_ref = Ref(n_checked)
    check_grads_recursive!(g_std[1].base.pair_rep_builder, g_gpu[1].base.pair_rep_builder, mr_ref, nc_ref)
    max_rel = mr_ref[]
    n_checked = nc_ref[]
end

pass = max_rel < 0.02
@printf("  Checked %d parameter gradients\n", n_checked)
@printf("  Max relative error: %.6f %s\n", max_rel, pass ? "PASS" : "FAIL")

# 3. Performance comparison
println("\n--- Performance ---")
function bench(name, f, n_warmup=3, n_iter=10)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:n_iter; f(); end
    return t / n_iter * 1000
end

t_sc_std = bench("SC fwd std", () -> forward_branching_from_raw_features(model, raw_features))
t_sc_gpu = bench("SC fwd gpu", () -> forward_branching_from_raw_features_gpu(model, raw_features))

t_train_std = bench("train std", () -> begin
    Flux.withgradient(model) do m
        out = forward_branching_from_raw_features(m, raw_features)
        sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v]) + sum(out[:split]) + sum(out[:del])
    end
end)

t_train_gpu = bench("train gpu", () -> begin
    Flux.withgradient(model) do m
        out = forward_branching_from_raw_features_gpu(m, raw_features)
        sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v]) + sum(out[:split]) + sum(out[:del])
    end
end)

t_full_std = bench("full std", () -> begin
    forward_branching_from_raw_features(model, raw_features)
    Flux.withgradient(model) do m
        out = forward_branching_from_raw_features(m, raw_features)
        sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v]) + sum(out[:split]) + sum(out[:del])
    end
end)

t_full_gpu = bench("full gpu", () -> begin
    forward_branching_from_raw_features_gpu(model, raw_features)
    Flux.withgradient(model) do m
        out = forward_branching_from_raw_features_gpu(m, raw_features)
        sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v]) + sum(out[:split]) + sum(out[:del])
    end
end)

println()
@printf("  %-40s %8s %8s %8s\n", "Operation", "Standard", "GPU-opt", "Speedup")
@printf("  %-40s %8s %8s %8s\n", "-"^40, "-"^8, "-"^8, "-"^7)
@printf("  %-40s %7.1f ms %7.1f ms %6.2fx\n", "Self-conditioning forward", t_sc_std, t_sc_gpu, t_sc_std / t_sc_gpu)
@printf("  %-40s %7.1f ms %7.1f ms %6.2fx\n", "Training forward+backward", t_train_std, t_train_gpu, t_train_std / t_train_gpu)
@printf("  %-40s %7.1f ms %7.1f ms %6.2fx\n", "Full training step", t_full_std, t_full_gpu, t_full_std / t_full_gpu)

println()
hours_std = t_full_std * 20000 / 1000 / 3600
hours_gpu = t_full_gpu * 20000 / 1000 / 3600
@printf("  20k training iterations:\n")
@printf("    Standard:      %.1f hours\n", hours_std)
@printf("    GPU-optimized: %.1f hours\n", hours_gpu)
@printf("    Time saved:    %.1f hours (%.0f%%)\n", hours_std - hours_gpu, (1 - hours_gpu/hours_std) * 100)

println("\nDone!")
