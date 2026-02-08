#!/usr/bin/env julia
# Sample from branching model to test GPU mode correctness.
#
# Supports all three GPU modes via --mode flag:
#   cutile        (default) — cuTile fused kernels
#   nocutile      — NNlib attention + CuArray layer overrides
#   no_overrides  — TF32 only, no CuArray overrides
#
# IMPORTANT: Switching modes requires clearing precompile cache:
#   rm -rf ~/.julia/compiled/v1.12/LaProteina/
#
# Usage:
#   julia -t 4 scripts/sample_branching_gpu_test.jl --mode cutile
#   julia -t 4 scripts/sample_branching_gpu_test.jl --mode nocutile
#   LAPROTEINA_NOCUTILE=1 julia -t 4 scripts/sample_branching_gpu_test.jl --mode nocutile

# ── Parse CLI args before anything else (env vars must be set before `using`) ──
mode = "cutile"
weights_path = nothing
n_samples = 5
for i in eachindex(ARGS)
    if ARGS[i] == "--mode" && i < length(ARGS)
        global mode = ARGS[i+1]
    elseif ARGS[i] == "--weights" && i < length(ARGS)
        global weights_path = ARGS[i+1]
    elseif ARGS[i] == "--n" && i < length(ARGS)
        global n_samples = parse(Int, ARGS[i+1])
    end
end

if mode == "nocutile"
    ENV["LAPROTEINA_NOCUTILE"] = "1"
    ENV["LAPROTEINA_NO_OVERRIDES"] = ""
elseif mode == "no_overrides"
    ENV["LAPROTEINA_NO_OVERRIDES"] = "1"
    ENV["LAPROTEINA_NOCUTILE"] = ""
elseif mode == "cutile"
    delete!(ENV, "LAPROTEINA_NOCUTILE")
    delete!(ENV, "LAPROTEINA_NO_OVERRIDES")
else
    error("Unknown mode: $mode. Use: cutile, nocutile, no_overrides")
end

t_wall_start = time()

# ── Now activate and load packages ──
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using LaProteina: DecoderTransformer, load_decoder_weights!, samples_to_pdb
using BranchingFlows
using BranchingFlows: BranchingState, CoalescentFlow
using ForwardBackward: ContinuousState, DiscreteState, tensor
using Flowfusion: MaskedState, RDNFlow
import Flowfusion
using Flux: cpu, gpu
using CUDA
using Distributions: Beta, Poisson
using Random
using Statistics
using JLD2

include(joinpath(@__DIR__, "..", "src", "branching", "branching_score_network.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_inference.jl"))

Random.seed!(42)

println("=" ^ 70)
println("Branching Full Model Sampling - GPU Mode Test")
println("=" ^ 70)
println("Mode: $mode")
println("cuTile available: $(LaProteina._HAS_CUTILE)")
println("No overrides: $(LaProteina._NO_OVERRIDES)")
println("NoCuTile forced: $(LaProteina._FORCE_NOCUTILE)")

dev = CUDA.functional() ? gpu : identity
println("Device: $(CUDA.functional() ? "GPU" : "CPU")")

# ── Load model ──
weights_dir = joinpath(@__DIR__, "..", "weights")

# Default: use good_runs weights
if weights_path === nothing
    weights_path = joinpath(@__DIR__, "..", "good_runs",
        "bf_20k_beta12_cascale2_softclamp35", "branching_full_final.jld2")
end
println("Weights: $weights_path")

base = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=256, latent_dim=8, output_param=:v,
    qk_ln=true, update_pair_repr=false
)
model = BranchingScoreNetwork(base)

weights = load(weights_path)
if haskey(weights, "base")
    Flux.loadmodel!(model.base, weights["base"])
    println("  Loaded fine-tuned base weights")
else
    load_score_network_weights!(model.base, joinpath(weights_dir, "score_network.npz"))
    println("  Loaded pretrained base weights (no fine-tuned base in checkpoint)")
end
if haskey(weights, "indel_time_proj")
    Flux.loadmodel!(model.indel_time_proj, weights["indel_time_proj"])
end
if haskey(weights, "split_head")
    Flux.loadmodel!(model.split_head, weights["split_head"])
end
if haskey(weights, "del_head")
    Flux.loadmodel!(model.del_head, weights["del_head"])
end
model = dev(model)
println("Model loaded")

# ── Load decoder ──
decoder = DecoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=8, qk_ln=true, update_pair_repr=false
)
load_decoder_weights!(decoder, joinpath(weights_dir, "decoder.npz"))
println("Decoder loaded (CPU)")

output_dir = joinpath(@__DIR__, "..", "test", "samples_branching_gpu_test_$(mode)")
mkpath(output_dir)
println("Output dir: $output_dir")

t_load_done = time()
load_time = t_load_done - t_wall_start
println("Load/compile time: $(round(load_time, digits=1))s")

latent_dim = 8

# Cosine time schedule
step_func(t) = Float32(1 - (cos(t * pi) + 1) / 2)
step_number = 500
steps = step_func.(0f0:Float32(1/step_number):1f0)
nsteps = length(steps) - 1

println("Steps: $nsteps (cosine schedule)")
println()

# Timing accumulators
sample_times = Float64[]          # generation time per sample
decode_times = Float64[]          # decode+save time per sample
sample_lengths = Int[]            # final lengths
sample_status = String[]          # "ok" or "FAILED"

for sample_idx in 1:n_samples
    println("--- Sample $sample_idx/$n_samples ---")

    initial_length = max(1, rand(Poisson(100)))

    P = create_branching_processes(; latent_dim=latent_dim)
    P_ca = P.P[1]
    P_ll = P.P[2]

    wrapper = BranchingScoreNetworkWrapper(dev(model), latent_dim;
        self_cond=true, dev=dev, processes=(P_ca, P_ll))

    X0 = create_initial_state(P_ca, P_ll, initial_length, latent_dim)

    Xt = X0
    t_gen_start = time()
    errored = false
    last_step = 0
    for i in 1:nsteps
        t1, t2 = steps[i], steps[i+1]
        local hat
        try
            hat = wrapper(t1, Xt)
        catch e
            println("  ERROR at step $i (t=$t1): ", sprint(showerror, e; context=:limit=>true)[1:min(500, end)])
            errored = true
            break
        end
        Xt = Flowfusion.step(P, Xt, hat, t1, t2)
        last_step = i

        L_current = size(Xt.groupings, 1)
        if i % 100 == 0
            step_elapsed = time() - t_gen_start
            println("  Step $i/$nsteps: t=$(round(t2, digits=3)), L=$L_current, elapsed=$(round(step_elapsed, digits=1))s")
        end
    end
    gen_elapsed = time() - t_gen_start

    if errored
        push!(sample_times, gen_elapsed)
        push!(decode_times, 0.0)
        push!(sample_lengths, -1)
        push!(sample_status, "FAILED@step$last_step")
        println("  FAILED after $(round(gen_elapsed, digits=1))s at step $last_step")
        println()
        continue
    end

    # Extract final state
    ca_tensor = tensor(Xt.state[1].S)
    ll_tensor = tensor(Xt.state[2].S)
    ca_coords = dropdims(ca_tensor, dims=3)
    latents = dropdims(ll_tensor, dims=3)
    final_L = size(ca_coords, 2)

    println("  Generated: L=$initial_length -> $final_L in $(round(gen_elapsed, digits=1))s ($(round(gen_elapsed/nsteps*1000, digits=1))ms/step)")

    # CA-CA distances
    dists = [sqrt(sum((ca_coords[:, i+1] .- ca_coords[:, i]).^2)) for i in 1:(final_L-1)]
    mean_d = mean(dists)
    println("  Mean CA-CA: $(round(mean_d, digits=3)) nm")

    # Decode and save
    t_dec_start = time()
    ca_3d = reshape(ca_coords, 3, final_L, 1)
    ll_3d = reshape(latents, latent_dim, final_L, 1)
    mask = ones(Float32, final_L, 1)

    dec_input = Dict(:z_latent => ll_3d, :ca_coors => ca_3d, :mask => mask)
    dec_out = decoder(dec_input)

    samples = Dict(
        :ca_coords => ca_3d,
        :latents => ll_3d,
        :all_atom_coords => dec_out[:coors],
        :aatype => dec_out[:aatype_max],
        :atom_mask => dec_out[:atom_mask],
        :mask => mask
    )

    samples_to_pdb(samples, output_dir; prefix="$(mode)_$sample_idx", save_all_atom=true)
    dec_elapsed = time() - t_dec_start

    aatype = dec_out[:aatype_max][:, 1]
    seq = join([index_to_aa(aa) for aa in aatype])
    println("  Decoded+saved in $(round(dec_elapsed, digits=1))s | Seq: $(seq[1:min(40, length(seq))])...")
    println()

    push!(sample_times, gen_elapsed)
    push!(decode_times, dec_elapsed)
    push!(sample_lengths, final_L)
    push!(sample_status, "ok")
end

total_time = time() - t_wall_start
gen_total = sum(sample_times)
dec_total = sum(decode_times)
n_ok = count(==("ok"), sample_status)

println("=" ^ 70)
println("TIMING SUMMARY — mode: $mode")
println("=" ^ 70)
println("  Load/compile:  $(round(load_time, digits=1))s")
println("  Generation:    $(round(gen_total, digits=1))s total, $(n_ok > 0 ? round(gen_total/n_ok, digits=1) : "N/A")s/sample avg")
println("  Decode+save:   $(round(dec_total, digits=1))s total")
println("  Wall clock:    $(round(total_time, digits=1))s")
println()
println("  Per-sample breakdown:")
for i in 1:length(sample_times)
    st = sample_status[i]
    L = sample_lengths[i]
    gt = round(sample_times[i], digits=1)
    dt = round(decode_times[i], digits=1)
    println("    #$i: gen=$(gt)s, decode=$(dt)s, L=$L, status=$st")
end
println()
println("Samples saved to: $output_dir")
println("=" ^ 70)
