#!/usr/bin/env julia
# Sample from the triangle-update branching model (OUBridgeExpVar processes)
# Same as sample_branching_full_OU.jl but with update_pair_repr=true (LD2 base).
#
# Usage:
#   julia --project=/home/claudey/JuProteina/run -t 4 scripts/sample_branching_full_OU_tri.jl --compare 5 --ca-v0 40 --ll-v0 40
#   julia --project=/home/claudey/JuProteina/run -t 4 scripts/sample_branching_full_OU_tri.jl --weights /path/to/checkpoint.jld2

using LaProteina
using OnionTile  # Activates cuTile CuArray overrides for Onion dispatch hooks
using LaProteina: DecoderTransformer, load_decoder_weights_st!, samples_to_pdb
using BranchingFlows
using BranchingFlows: BranchingState, CoalescentFlow
using ForwardBackward: ContinuousState, DiscreteState, tensor, OUBridgeExpVar
using Flowfusion: MaskedState
import Flowfusion
using Flux
using Flux: cpu, gpu
using CUDA
using Distributions: Beta, Poisson
using Random
using Statistics
using JLD2

use_annealed = "--annealed" in ARGS
function parse_arg(args, flag)
    for i in 1:length(args)-1
        if args[i] == flag
            return parse(Float32, args[i+1])
        end
    end
    return nothing
end
function parse_string_arg(args, flag)
    for i in 1:length(args)-1
        if args[i] == flag
            return args[i+1]
        end
    end
    return nothing
end
ca_v0_override = parse_arg(ARGS, "--ca-v0")
ll_v0_override = parse_arg(ARGS, "--ll-v0")
compare_n = let v = parse_arg(ARGS, "--compare"); v === nothing ? nothing : Int(v) end
start_length_param = let v = parse_arg(ARGS, "--start-length"); v === nothing ? 0 : Int(v) end
weights_override = parse_string_arg(ARGS, "--weights")

Random.seed!(42)

println("=" ^ 70)
println("Branching Full Model Sampling - Triangle Updates - OUBridgeExpVar - Cosine Steps")
println("=" ^ 70)

dev = CUDA.functional() ? gpu : identity
println("Device: $(CUDA.functional() ? "GPU" : "CPU")")

# Load model — triangle update variant (LD2 base architecture)
base = ScoreNetwork(
    n_layers=14, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=256, latent_dim=8, output_param=:v,
    qk_ln=true, update_pair_repr=true, update_pair_every_n=2,
    use_tri_mult=true
)
model = BranchingScoreNetwork(base)

# Load weights from checkpoint
full_weights_path = if weights_override !== nothing
    weights_override
else
    # Default: most recent triangle training checkpoint
    "/home/claudey/JuProteina/LaProteina.jl/outputs/branching_OU_tri_20260222_115301/checkpoints/checkpoint_batch025000.jld2"
end
println("Loading weights from: $full_weights_path")
weights = load(full_weights_path)
if haskey(weights, "base")
    Flux.loadmodel!(model.base, weights["base"]; strict=false)
    println("  Loaded fine-tuned base weights")
else
    error("Checkpoint does not contain base weights: $full_weights_path")
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

# Load decoder
decoder = DecoderTransformer(
    n_layers=12, token_dim=768, pair_dim=256, n_heads=12,
    dim_cond=128, latent_dim=8, qk_ln=true, update_pair_repr=false
)
load_decoder_weights_st!(decoder, "AE1_ucond_512.safetensors")
println("Decoder loaded (CPU)")

# ── Process configurations ──────────────────────────────────────────────
function make_ca_process(v0::Float32)
    OUBridgeExpVar(100f0, v0, 0.000000001f0, dec = -3f0)
end

function make_ll_process(v0::Float32)
    OUBridgeExpVar(100f0, v0, 0.000000001f0, dec = -0.1f0)
end

function ca_label_str(v0)
    v0 == 150f0 ? "ORIGINAL v0=150" : "v0=$v0 ($(round(sqrt(v0/150f0), digits=2))× noise amplitude)"
end

function ll_label_str(v0)
    v0 == 50f0 ? "ORIGINAL v0=50" : "v0=$v0 ($(round(sqrt(v0/50f0), digits=2))× noise amplitude)"
end

mod_ca_v0 = if ca_v0_override !== nothing
    ca_v0_override
elseif use_annealed
    37.5f0
else
    150f0
end

mod_ll_v0 = if ll_v0_override !== nothing
    ll_v0_override
elseif use_annealed
    12.5f0
else
    50f0
end

latent_dim = 8

println("Initial length distribution: Poisson($start_length_param) + 1")

# Cosine time schedule
step_func(t) = Float32(1 - (cos(t * pi) + 1) / 2)
step_number = 500
steps = step_func.(0f0:Float32(1/step_number):1f0)
nsteps = length(steps) - 1

println("Steps: $nsteps (cosine schedule)")
println("First 5 steps: $(round.(steps[1:5], digits=4))")
println("Last 5 steps: $(round.(steps[end-4:end], digits=4))")
println()

# ── Sampling function ───────────────────────────────────────────────────

function run_samples(n_samples, output_dir, prefix, P_flow, wrapper, decoder,
                     steps, nsteps, latent_dim; poisson_param=100)
    mkpath(output_dir)
    for sample_idx in 1:n_samples
        println("--- $prefix sample $sample_idx ---")

        initial_length = 1 + rand(Poisson(poisson_param))
        reset_self_conditioning!(wrapper)
        X0 = create_initial_state(initial_length, latent_dim)

        Xt = X0
        for i in 1:nsteps
            t1, t2 = steps[i], steps[i+1]

            L_current = size(Xt.groupings, 1)
            if L_current == 0
                println("  WARNING: Protein reached L=0 at step $i, skipping")
                break
            elseif L_current > 400
                println("  WARNING: Protein grew to L=$L_current at step $i (>400), skipping to avoid OOM")
                break
            end

            hat = wrapper(t1, Xt)
            Xt = Flowfusion.step(P_flow, Xt, hat, t1, t2)

            L_current = size(Xt.groupings, 1)
            if i % 100 == 0
                println("  Step $i/$nsteps: t=$(round(t2, digits=3)), L=$L_current")
            end
        end

        ca_tensor = tensor(Xt.state[1].S)
        ll_tensor = tensor(Xt.state[2].S)
        ca_coords = dropdims(ca_tensor, dims=3)
        latents = dropdims(ll_tensor, dims=3)
        final_L = size(ca_coords, 2)

        println("  Final length: $final_L (started at $initial_length)")

        if final_L == 0
            println("  Skipping empty protein")
            println()
            continue
        end

        dists = [sqrt(sum((ca_coords[:, i+1] .- ca_coords[:, i]).^2)) for i in 1:(final_L-1)]
        mean_d = final_L > 1 ? mean(dists) : 0.0
        println("  Mean CA-CA: $(round(mean_d, digits=3)) nm")

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

        pdb_prefix = "$(prefix)_$(sample_idx)"
        samples_to_pdb(samples, output_dir; prefix=pdb_prefix, save_all_atom=true)

        aatype = dec_out[:aatype_max][:, 1]
        seq = join([index_to_aa(aa) for aa in aatype])
        println("  Sequence: $(seq[1:min(40, length(seq))])...")
        println()

        GC.gc()
        CUDA.reclaim()
    end
end

# ── Main ────────────────────────────────────────────────────────────────

P_idx = NullProcess()
branch_time_dist = Beta(1.0, 2.0)

wrapper = BranchingScoreNetworkWrapper(model, latent_dim;
    self_cond=true, dev=dev, processes=nothing)

if compare_n !== nothing
    n = compare_n

    parts = String[]
    ca_v0_override !== nothing && push!(parts, "ca$(Int(mod_ca_v0))")
    ll_v0_override !== nothing && push!(parts, "ll$(Int(mod_ll_v0))")
    use_annealed && isempty(parts) && push!(parts, "annealed")
    tag = isempty(parts) ? "original" : join(parts, "_")
    output_dir = joinpath("/home/claudey/JuProteina/inference_outputs", "branching_tri_compare_$(tag)_n$(n)")

    # Original process
    P_orig = CoalescentFlow((make_ca_process(150f0), make_ll_process(50f0), P_idx), branch_time_dist)
    println("── Original process ($n samples) ──")
    println("  CA: $(ca_label_str(150f0))  LL: $(ll_label_str(50f0))")
    run_samples(n, output_dir, "original", P_orig, wrapper, decoder,
                steps, nsteps, latent_dim; poisson_param=start_length_param)

    # Modified process
    P_mod = CoalescentFlow((make_ca_process(mod_ca_v0), make_ll_process(mod_ll_v0), P_idx), branch_time_dist)
    println("── Modified process ($n samples) ──")
    println("  CA: $(ca_label_str(mod_ca_v0))  LL: $(ll_label_str(mod_ll_v0))")
    run_samples(n, output_dir, "modified", P_mod, wrapper, decoder,
                steps, nsteps, latent_dim; poisson_param=start_length_param)

    println("=" ^ 70)
    println("Comparison samples saved to: $output_dir")
    println("=" ^ 70)
else
    P = CoalescentFlow((make_ca_process(mod_ca_v0), make_ll_process(mod_ll_v0), P_idx), branch_time_dist)
    println("CA process: $(ca_label_str(mod_ca_v0))  (θ=100, v1=1e-9, dec=-3)")
    println("LL process: $(ll_label_str(mod_ll_v0))  (θ=100, v1=1e-9, dec=-0.1)")

    has_custom = ca_v0_override !== nothing || ll_v0_override !== nothing
    output_suffix = if has_custom
        parts = String[]
        ca_v0_override !== nothing && push!(parts, "ca$(Int(mod_ca_v0))")
        ll_v0_override !== nothing && push!(parts, "ll$(Int(mod_ll_v0))")
        "_" * join(parts, "_")
    elseif use_annealed
        "_annealed"
    else
        ""
    end
    output_dir = joinpath("/home/claudey/JuProteina/inference_outputs", "branching_tri_full_OU$output_suffix")

    n_samples = 10
    run_samples(n_samples, output_dir, "OU_cosine", P, wrapper, decoder,
                steps, nsteps, latent_dim; poisson_param=start_length_param)

    println("=" ^ 70)
    println("Samples saved to: $output_dir")
    println("=" ^ 70)
end
