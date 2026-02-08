#!/usr/bin/env julia
# Profile the feature projection part of forward_from_raw_features_gpu
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

function bench(name, f, n_warmup=5, n_iter=20)
    for _ in 1:n_warmup; f(); end
    CUDA.synchronize()
    t = CUDA.@elapsed for _ in 1:n_iter; f(); end
    @printf("  %-55s %.2f ms\n", name, t / n_iter * 1000)
end

mask = raw_features.mask
L2, B2 = size(mask)
mask_exp = reshape(mask, 1, L2, B2)
pair_mask = reshape(mask, L2, 1, B2) .* reshape(mask, 1, L2, B2)
pair_mask_exp = reshape(pair_mask, 1, L2, L2, B2)

println("="^70)
println("Feature Projection Profiling")
println("="^70)

println("\n--- Forward projections ---")
bench("cond_factory.projection", () -> model.cond_factory.projection(raw_features.cond_raw))
bench("init_repr_factory.projection", () -> model.init_repr_factory.projection(raw_features.seq_raw))
bench("pair_rep_builder projection", () -> model.pair_rep_builder.init_repr_factory.projection(raw_features.pair_raw))
bench("pair_cond projection", () -> model.pair_rep_builder.cond_factory.projection(raw_features.pair_cond_raw))

println("\n--- Conditioning transitions ---")
cond = model.cond_factory.projection(raw_features.cond_raw) .* mask_exp
bench("transition_c_1", () -> model.transition_c_1(cond, mask))
bench("transition_c_2", () -> model.transition_c_2(cond, mask))

println("\n--- Pair AdaLN ---")
pair_rep = model.pair_rep_builder.init_repr_factory.projection(raw_features.pair_raw) .* pair_mask_exp
pair_cond = model.pair_rep_builder.cond_factory.projection(raw_features.pair_cond_raw) .* pair_mask_exp
bench("pair_rep_builder.adaln", () -> model.pair_rep_builder.adaln(pair_rep, pair_cond, pair_mask))

println("\n--- Pair pre-normalization ---")
first_pba = model.transformer_layers[1].mha.mha
pair_eps = first_pba.pair_norm.ϵ
bench("pytorch_normalise (pair)", () -> LaProteina.pytorch_normalise(pair_rep; dims=1, eps=pair_eps))

println("\n--- Backward of full feature projection ---")
bench("Feature projection backward", () -> begin
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
        pc = m.pair_rep_builder.cond_factory.projection(raw_features.pair_cond_raw) .* pme
        pr = m.pair_rep_builder.adaln(pr, pc, pm)
        pn = LaProteina.pytorch_normalise(pr; dims=1, eps=pair_eps)
        sum(cond2) + sum(seqs2) + sum(pr) + sum(pn)
    end, model)
end)

println("\n--- Total forward_from_raw_features_gpu ---")
bench("Full GPU forward", () -> begin
    out = forward_from_raw_features_gpu(model, raw_features)
    nothing
end)
bench("Full GPU backward", () -> begin
    Zygote.gradient(m -> begin
        out = forward_from_raw_features_gpu(m, raw_features)
        sum(out[:bb_ca][:v]) + sum(out[:local_latents][:v])
    end, model)
end)

println("\nDone!")
