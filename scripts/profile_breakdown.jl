#!/usr/bin/env julia
# Detailed breakdown of remaining GPU-optimized costs
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using CUDA
using Flux
using Flux: Zygote
using Printf

LaProteina.enable_tf32!()

L = 128; B = 4

model = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256,
    n_heads=12, dim_cond=256, qk_ln=true,
    update_pair_repr=false
) |> gpu

seq_raw_dim = size(model.init_repr_factory.projection.weight, 2)
cond_raw_dim = size(model.cond_factory.projection.weight, 2)
pair_raw_dim = size(model.pair_rep_builder.init_repr_factory.projection.weight, 2)
pair_cond_raw_dim = size(model.pair_rep_builder.cond_factory.projection.weight, 2)

raw_features = ScoreNetworkRawFeatures(
    CUDA.randn(Float32, seq_raw_dim, L, B),
    CUDA.randn(Float32, cond_raw_dim, L, B),
    CUDA.randn(Float32, pair_raw_dim, L, L, B),
    CUDA.randn(Float32, pair_cond_raw_dim, L, L, B),
    CUDA.ones(Float32, L, B)
)

function bench(name, f, n_warmup=3, n_iter=10)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:n_iter; f(); end
    @printf("  %-55s %.2f ms\n", name, t / n_iter * 1000)
    return t / n_iter * 1000
end

println("="^70)
println("GPU-Optimized Cost Breakdown (L=$L, B=$B)")
println("="^70)

# Pre-compute things needed for block profiling
mask = raw_features.mask
x = CUDA.randn(Float32, 768, L, B)
pair = CUDA.randn(Float32, 256, L, L, B)
cond = CUDA.randn(Float32, 256, L, B)
first_pba = model.transformer_layers[1].mha.mha
pair_eps = first_pba.pair_norm.ϵ
pair_normed = LaProteina.pytorch_normalise(pair; dims=1, eps=pair_eps)

println("\n--- Feature Projection (forward) ---")
bench("pair_raw projection", () -> model.pair_rep_builder.init_repr_factory.projection(raw_features.pair_raw))
bench("pair_cond batch projection", () -> begin
    pcbr = raw_features.pair_cond_raw[:, 1, 1, :]
    model.pair_rep_builder.cond_factory.projection(pcbr)
end)
bench("pair_normalise", () -> LaProteina.pytorch_normalise(pair; dims=1, eps=pair_eps))

println("\n--- Feature Projection (backward) ---")
bench("Feature projection backward (GPU)", () -> begin
    Zygote.gradient(m -> begin
        mask2 = raw_features.mask
        L3, B3 = size(mask2)
        me = reshape(mask2, 1, L3, B3)
        pm = reshape(mask2, L3, 1, B3) .* reshape(mask2, 1, L3, B3)
        pme = reshape(pm, 1, L3, L3, B3)

        cond2 = m.cond_factory.projection(raw_features.cond_raw) .* me
        seqs2 = m.init_repr_factory.projection(raw_features.seq_raw) .* me
        pr = m.pair_rep_builder.init_repr_factory.projection(raw_features.pair_raw) .* pme
        cond2 = m.transition_c_1(cond2, mask2)
        cond2 = m.transition_c_2(cond2, mask2)
        # Batch-level pair conditioning
        pcbr = raw_features.pair_cond_raw[:, 1, 1, :]
        pc = m.pair_rep_builder.cond_factory.projection(pcbr)
        if m.pair_rep_builder.cond_factory.use_ln && !isnothing(m.pair_rep_builder.cond_factory.ln)
            pc = m.pair_rep_builder.cond_factory.ln(pc)
        end
        pr = m.pair_rep_builder.adaln(pr, pc, pm)
        pn = LaProteina.pytorch_normalise(pr; dims=1, eps=pair_eps)
        sum(cond2) + sum(seqs2) + sum(pr) + sum(pn)
    end, model)
end)

println("\n--- Per-block costs (backward) ---")
for n in [1, 7, 14]
    bench("$n blocks prenorm backward", () -> begin
        blocks = model.transformer_layers[1:n]
        Zygote.gradient(x -> begin
            y = x
            for i in 1:n
                y = LaProteina._transformer_block_prenormed(blocks[i], y, pair, pair_normed, cond, mask)
            end
            sum(y)
        end, x)
    end)
end

println("\n--- Full forward+backward ---")
t_fwd = bench("Full GPU forward", () -> forward_from_raw_features_gpu(model, raw_features))
t_bwd = bench("Full GPU backward", () -> begin
    Zygote.gradient(m -> begin
        out = forward_from_raw_features_gpu(m, raw_features)
        sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v])
    end, model)
end)

@printf("\n  Forward:  %.1f ms\n", t_fwd)
@printf("  Backward: %.1f ms\n", t_bwd)
@printf("  Estimated per-block backward (from 14-block): %.1f ms\n", 0.0)

println("\nDone!")
