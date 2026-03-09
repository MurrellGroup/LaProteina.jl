#!/usr/bin/env julia
# Sample from the triangle-update branching model (OUBridgeExpVar processes)
# Same as sample_branching_full_OU.jl but with update_pair_repr=true (LD2 base).
#
# Usage:
#   julia --project=/home/claudey/JuProteina/run -t 4 scripts/sample_branching_full_OU_tri.jl --compare 5 --ca-v0 40 --ll-v0 40
#   julia --project=/home/claudey/JuProteina/run -t 4 scripts/sample_branching_full_OU_tri.jl --weights /path/to/checkpoint.jld2
#   julia --project=/home/claudey/JuProteina/run -t 4 scripts/sample_branching_full_OU_tri.jl --ca-v0 60 --ll-v0 40 --n-samples 20 --trajectory --output-dir /path/to/output
#   julia --project=/home/claudey/JuProteina/run -t 4 scripts/sample_branching_full_OU_tri.jl --ca-v0 40 --ll-v0 40 --seed 123

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
save_trajectory = "--trajectory" in ARGS
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
n_samples_override = let v = parse_arg(ARGS, "--n-samples"); v === nothing ? nothing : Int(v) end
output_dir_override = parse_string_arg(ARGS, "--output-dir")
weights_override = parse_string_arg(ARGS, "--weights")
seed_val = let v = parse_arg(ARGS, "--seed"); v === nothing ? 42 : Int(v) end

Random.seed!(seed_val)

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
                     steps, nsteps, latent_dim; poisson_param=100,
                     save_trajectory::Bool=false)
    mkpath(output_dir)
    for sample_idx in 1:n_samples
        println("--- $prefix sample $sample_idx ---")

        initial_length = 1 + rand(Poisson(poisson_param))
        reset_self_conditioning!(wrapper)
        X0 = create_initial_state(initial_length, latent_dim)

        # Trajectory recording
        traj_frames = save_trajectory ? Vector{Dict{Symbol,Any}}() : nothing

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

            # Record trajectory frame with all-atom decoder outputs
            if save_trajectory
                xt_ca = dropdims(tensor(Xt.state[1].S), dims=3)   # [3, L]
                xt_ll = dropdims(tensor(Xt.state[2].S), dims=3)   # [latent_dim, L]
                x1hat_ca = dropdims(tensor(hat[1][1]), dims=3)     # [3, L]
                x1hat_ll = dropdims(tensor(hat[1][2]), dims=3)     # [latent_dim, L]

                frame = Dict{Symbol,Any}(
                    :xt_ca => Array(xt_ca),
                    :x1hat_ca => Array(x1hat_ca),
                    :t => Float32(t1),
                    :step => i,
                    :L => L_current
                )

                # Decode x1hat → all-atom (every step)
                x1h_ca3 = reshape(Array(x1hat_ca), 3, L_current, 1)
                x1h_ll3 = reshape(Array(x1hat_ll), latent_dim, L_current, 1)
                msk = ones(Float32, L_current, 1)
                d1 = decoder(Dict(:z_latent => x1h_ll3, :ca_coors => x1h_ca3, :mask => msk))
                frame[:x1hat_all_atom]  = Array(d1[:coors][:,:,:,1])     # [3,37,L]
                frame[:x1hat_aatype]    = Array(d1[:aatype_max][:,1])    # [L]
                frame[:x1hat_atom_mask] = Array(d1[:atom_mask][:,:,1])   # [37,L]

                # Also decode Xt → all-atom (every step)
                xt_ca3 = reshape(Array(xt_ca), 3, L_current, 1)
                xt_ll3 = reshape(Array(xt_ll), latent_dim, L_current, 1)
                d0 = decoder(Dict(:z_latent => xt_ll3, :ca_coors => xt_ca3, :mask => msk))
                frame[:xt_all_atom]  = Array(d0[:coors][:,:,:,1])
                frame[:xt_aatype]    = Array(d0[:aatype_max][:,1])
                frame[:xt_atom_mask] = Array(d0[:atom_mask][:,:,1])

                push!(traj_frames, frame)

                if i % 100 == 0
                    println("    (decoder done for step $i)")
                end
            end

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

        # Save trajectory JLD2 if requested
        if save_trajectory && traj_frames !== nothing && !isempty(traj_frames)
            traj_path = joinpath(output_dir, "$(pdb_prefix)_trajectory.jld2")
            final_decoder = Dict{Symbol,Any}(
                :all_atom_coords => Array(dec_out[:coors][:, :, :, 1]),  # [3, 37, L]
                :aatype => Array(dec_out[:aatype_max][:, 1]),            # [L]
                :atom_mask => Array(dec_out[:atom_mask][:, :, 1]),       # [37, L]
                :ca_coords => Array(ca_coords)                          # [3, L]
            )
            metadata = Dict{String,Any}(
                "initial_length" => initial_length,
                "final_length" => final_L,
                "nsteps" => nsteps
            )
            jldsave(traj_path;
                frames=traj_frames,
                final_decoder=final_decoder,
                metadata=metadata
            )
            println("  Trajectory saved: $traj_path ($(length(traj_frames)) frames)")
        end

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
                steps, nsteps, latent_dim; poisson_param=start_length_param,
                save_trajectory=save_trajectory)

    # Modified process
    P_mod = CoalescentFlow((make_ca_process(mod_ca_v0), make_ll_process(mod_ll_v0), P_idx), branch_time_dist)
    println("── Modified process ($n samples) ──")
    println("  CA: $(ca_label_str(mod_ca_v0))  LL: $(ll_label_str(mod_ll_v0))")
    run_samples(n, output_dir, "modified", P_mod, wrapper, decoder,
                steps, nsteps, latent_dim; poisson_param=start_length_param,
                save_trajectory=save_trajectory)

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
    output_dir = if output_dir_override !== nothing
        output_dir_override
    else
        joinpath("/home/claudey/JuProteina/inference_outputs", "branching_tri_full_OU$output_suffix")
    end

    n_samples = n_samples_override !== nothing ? n_samples_override : 10
    run_samples(n_samples, output_dir, "OU_cosine", P, wrapper, decoder,
                steps, nsteps, latent_dim; poisson_param=start_length_param,
                save_trajectory=save_trajectory)

    println("=" ^ 70)
    println("Samples saved to: $output_dir")
    println("=" ^ 70)
end
