#!/usr/bin/env julia
# Diagnostic script: inspect deletion targets in branching_bridge training batches
# Purpose: Diagnose potential over-deletion in branching flows training

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LaProteina
using LaProteina: load_precomputed_shard
using BranchingFlows
using BranchingFlows: BranchingState, CoalescentFlow, branching_bridge, Deletion
using ForwardBackward: ContinuousState, DiscreteState, tensor
using Flowfusion: RDNFlow, MaskedState, floss, scalefloss, schedule_transform, element
using Distributions: Uniform, Beta, Poisson
using Statistics
using Random
using Printf
using JLD2

# Simple countmap replacement
function my_countmap(x)
    d = Dict{eltype(x), Int}()
    for v in x
        d[v] = get(d, v, 0) + 1
    end
    return d
end

# Include branching module (matching train_branching_full.jl)
include(joinpath(@__DIR__, "..", "src", "branching", "branching_score_network.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_states.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_training.jl"))
include(joinpath(@__DIR__, "..", "src", "branching", "branching_inference.jl"))

Random.seed!(42)

println("=" ^ 70)
println("DELETION TARGET DIAGNOSTIC")
println("=" ^ 70)

# ============================================================================
# Load one shard of training data
# ============================================================================
shard_path = expanduser("~/shared_data/afdb_laproteina/precomputed_shards/train_shard_01.jld2")
println("\nLoading shard: $shard_path")
proteins = load_precomputed_shard(shard_path)
println("Loaded $(length(proteins)) proteins")

# Show some protein stats
lengths = [sum(p.mask .> 0.5) for p in proteins]
println("Protein lengths: min=$(minimum(lengths)), max=$(maximum(lengths)), mean=$(round(mean(lengths), digits=1)), median=$(round(median(lengths), digits=1))")

# ============================================================================
# Create CoalescentFlow (matching train_branching_full.jl exactly)
# ============================================================================
latent_dim = 8
X0_mean_length = 100
deletion_pad = 1.1

P_ca = RDNFlow(3;
    zero_com=false,
    schedule=:log, schedule_param=2.0f0,
    sde_gt_mode=Symbol("1/t"), sde_gt_param=1.0f0,
    sc_scale_noise=0.1f0, sc_scale_score=1.0f0, t_lim_ode=0.98f0)
P_ll = RDNFlow(latent_dim;
    zero_com=false,
    schedule=:power, schedule_param=2.0f0,
    sde_gt_mode=:tan, sde_gt_param=1.0f0,
    sc_scale_noise=0.1f0, sc_scale_score=1.0f0, t_lim_ode=0.98f0)
branch_time_dist = Beta(2.0, 2.0)
P = CoalescentFlow((P_ca, P_ll), branch_time_dist)
println("CoalescentFlow created")

# X0 sampler (matching train_branching_full.jl exactly)
function X0_sampler_train(root)
    ca = ContinuousState(randn(Float32, 3, 1, 1))
    ll = ContinuousState(randn(Float32, latent_dim, 1, 1))
    return (ca, ll)
end

# protein_to_X1_simple (matching train_branching_full.jl exactly)
function protein_to_X1_simple(protein)
    ca_coords = protein.ca_coords
    z_mean = protein.z_mean
    z_log_scale = protein.z_log_scale
    mask = protein.mask
    L = length(mask)

    z_latent = z_mean .+ randn(Float32, size(z_mean)) .* exp.(z_log_scale)
    valid_mask = Vector{Bool}(mask .> 0.5)

    ca_state = MaskedState(
        ContinuousState(reshape(ca_coords, 3, 1, L)),
        valid_mask, valid_mask
    )
    latent_state = MaskedState(
        ContinuousState(reshape(z_latent, size(z_latent, 1), 1, L)),
        valid_mask, valid_mask
    )

    groupings = ones(Int, L)
    return BranchingState(
        (ca_state, latent_state),
        groupings;
        flowmask = valid_mask,
        branchmask = valid_mask
    )
end

# ============================================================================
# Create training batches and inspect deletion targets
# ============================================================================
println("\n" * "=" ^ 70)
println("CREATING TRAINING BATCHES")
println("=" ^ 70)

batch_size = 4
n_batches = 5

for batch_num in 1:n_batches
    println("\n" * "-" ^ 70)
    println("BATCH $batch_num")
    println("-" ^ 70)

    # Sample random proteins
    indices = rand(1:length(proteins), batch_size)
    X1s = [protein_to_X1_simple(proteins[i]) for i in indices]

    # Print X1 info
    println("\n  X1 (data) info:")
    for (b, idx) in enumerate(indices)
        x1 = X1s[b]
        L = length(x1.groupings)
        n_valid = sum(x1.padmask)
        n_flow = sum(x1.flowmask)
        n_branch = sum(x1.branchmask)
        n_del = sum(x1.del)
        println("    Sample $b (protein #$idx): L=$L, valid=$n_valid, flow=$n_flow, branch=$n_branch, del=$n_del (original protein length=$(Int(sum(proteins[idx].mask .> 0.5))))")
    end

    # Create the batch using branching_bridge
    t_dist = Uniform(0f0, 1f0)
    batch = branching_bridge(
        P, X0_sampler_train, X1s, t_dist;
        coalescence_factor = 1.0,
        use_branching_time_prob = 0.5,
        length_mins = Poisson(X0_mean_length),
        deletion_pad = deletion_pad
    )

    # ===== Batch shapes =====
    println("\n  Batch shapes:")
    del = batch.del
    padmask = batch.Xt.padmask
    branchmask = batch.Xt.branchmask
    flowmask = batch.Xt.flowmask
    groupings = batch.Xt.groupings
    descendants = batch.descendants
    splits_target = batch.splits_target

    L_batch, B = size(del)
    println("    del:          $(size(del))")
    println("    padmask:      $(size(padmask))")
    println("    branchmask:   $(size(branchmask))")
    println("    flowmask:     $(size(flowmask))")
    println("    groupings:    $(size(groupings))")
    println("    descendants:  $(size(descendants))")
    println("    splits_target:$(size(splits_target))")

    # ===== Per-sample deletion analysis =====
    println("\n  Per-sample analysis:")
    for b in 1:B
        n_del_true = sum(del[:, b])
        n_del_false = sum(.!del[:, b])
        n_pad_true = sum(padmask[:, b])
        n_pad_false = sum(.!padmask[:, b])
        n_branch_true = sum(branchmask[:, b])
        n_branch_false = sum(.!branchmask[:, b])
        n_flow_true = sum(flowmask[:, b])

        # Fraction of padmask=true positions that have del=true
        valid_positions = padmask[:, b]
        del_among_valid = sum(del[:, b] .& valid_positions)
        frac_del_of_valid = n_pad_true > 0 ? del_among_valid / n_pad_true : 0.0

        # Fraction of branchmask=true positions that have del=true
        branch_positions = branchmask[:, b]
        del_among_branch = sum(del[:, b] .& branch_positions)
        frac_del_of_branch = n_branch_true > 0 ? del_among_branch / n_branch_true : 0.0

        # Combined mask (what training loss uses)
        combined = padmask[:, b] .& branchmask[:, b]
        n_combined = sum(combined)
        del_among_combined = sum(del[:, b] .& combined)
        frac_del_of_combined = n_combined > 0 ? del_among_combined / n_combined : 0.0

        # X1 length vs Xt length
        x1_len = Int(sum(proteins[indices[b]].mask .> 0.5))
        xt_len = n_pad_true

        println("    Sample $b:")
        println("      X1 len (original protein): $x1_len")
        println("      Xt len (padmask=true):     $xt_len  (padded to L_batch=$L_batch)")
        println("      del=true:  $n_del_true / $L_batch total positions")
        println("      del=false: $n_del_false / $L_batch total positions")
        println("      padmask=true:  $n_pad_true,  padmask=false: $n_pad_false")
        println("      branchmask=true: $n_branch_true, branchmask=false: $n_branch_false")
        println("      flowmask=true:  $n_flow_true")
        println("      del among valid (padmask=true):     $del_among_valid / $n_pad_true = $(round(frac_del_of_valid, digits=4))")
        println("      del among branchable:               $del_among_branch / $n_branch_true = $(round(frac_del_of_branch, digits=4))")
        println("      del among combined (pad & branch):  $del_among_combined / $n_combined = $(round(frac_del_of_combined, digits=4))")

        # Descendant counts distribution
        valid_desc = descendants[valid_positions, b]
        desc_counts = my_countmap(valid_desc)
        println("      descendant counts distribution: $desc_counts")

        # Splits target distribution
        valid_splits = splits_target[valid_positions, b]
        println("      splits_target: min=$(round(minimum(valid_splits), digits=3)), max=$(round(maximum(valid_splits), digits=3)), mean=$(round(mean(valid_splits), digits=3))")
        n_zero_splits = sum(valid_splits .== 0)
        n_nonzero_splits = sum(valid_splits .> 0)
        println("      splits_target==0: $n_zero_splits, splits_target>0: $n_nonzero_splits")
    end

    # ===== Time values =====
    println("\n  Time values (raw uniform):")
    for b in 1:B
        @printf("    Sample %d: t=%.4f\n", b, batch.t[b])
    end

    # ===== Aggregate deletion stats =====
    total_valid = sum(padmask)
    total_del_valid = sum(del .& padmask)
    overall_del_frac = total_valid > 0 ? total_del_valid / total_valid : 0.0
    println("\n  AGGREGATE for this batch:")
    println("    Total valid positions (padmask=true): $total_valid")
    println("    Total del=true among valid:           $total_del_valid")
    println("    Overall deletion fraction:            $(round(overall_del_frac, digits=4))")

    # Check: do padded positions (padmask=false) have del=true?
    padded_del = sum(del .& .!padmask)
    println("    del=true among padded (padmask=false): $padded_del  (should be 0)")

    # Check: relationship between del and flowmask
    del_not_flow = sum(del .& .!flowmask .& padmask)
    println("    del=true & flowmask=false among valid: $del_not_flow")

    # Check: relationship between del and branchmask
    del_not_branch = sum(del .& .!branchmask .& padmask)
    println("    del=true & branchmask=false among valid: $del_not_branch")

    # ===== What does the loss function see? =====
    println("\n  LOSS FUNCTION PERSPECTIVE (combined_mask = padmask .* branchmask):")
    combined_mask = Float32.(padmask) .* Float32.(branchmask)
    total_combined = sum(combined_mask)
    del_float = Float32.(del)

    del_target_in_loss = sum(del_float .* combined_mask)
    nondel_target_in_loss = sum((1f0 .- del_float) .* combined_mask)
    del_frac_in_loss = total_combined > 0 ? del_target_in_loss / total_combined : 0.0

    println("    Total positions in loss:   $(Int(total_combined))")
    println("    del=1 targets in loss:     $(Int(del_target_in_loss))")
    println("    del=0 targets in loss:     $(Int(nondel_target_in_loss))")
    println("    Fraction del=1 in loss:    $(round(del_frac_in_loss, digits=4))")
end

# ============================================================================
# Focused analysis: Effect of deletion_pad on deletion fractions
# ============================================================================
println("\n\n" * "=" ^ 70)
println("DELETION PAD SWEEP: Effect of deletion_pad on del fraction")
println("=" ^ 70)

# Use similar-length proteins to avoid negative Poisson issues
# Pick proteins close to X0_mean_length
similar_idx = findall(l -> 80 <= l <= 120, lengths)
println("\nUsing $(length(similar_idx)) proteins with lengths 80-120 for sweep")
test_indices = similar_idx[rand(1:length(similar_idx), batch_size)]
test_x1_lengths = [Int(sum(proteins[i].mask .> 0.5)) for i in test_indices]

println("Test proteins X1 lengths: $test_x1_lengths")
println("X0_mean_length: $X0_mean_length")

for dp in [0.0, 0.5, 1.0, 1.1, 1.5, 2.0]
    # Run multiple times to get an average
    del_fs_valid = Float64[]
    del_fs_combined = Float64[]
    n_success = 0
    for trial in 1:10
        Random.seed!(123 + trial)
        try
            X1s_copy = [protein_to_X1_simple(proteins[i]) for i in test_indices]
            batch = branching_bridge(
                P, X0_sampler_train, X1s_copy, Uniform(0f0, 1f0);
                coalescence_factor = 1.0,
                use_branching_time_prob = 0.5,
                length_mins = Poisson(X0_mean_length),
                deletion_pad = dp
            )

            del = batch.del
            padmask = batch.Xt.padmask
            branchmask = batch.Xt.branchmask
            L_batch, B = size(del)

            for b in 1:B
                valid = padmask[:, b]
                combined = valid .& branchmask[:, b]
                n_valid = sum(valid)
                n_combined = sum(combined)
                n_del_valid = sum(del[:, b] .& valid)
                n_del_combined = sum(del[:, b] .& combined)
                push!(del_fs_valid, n_valid > 0 ? n_del_valid / n_valid : 0.0)
                push!(del_fs_combined, n_combined > 0 ? n_del_combined / n_combined : 0.0)
            end
            n_success += 1
        catch e
            # Skip failures (negative Poisson lambda)
        end
    end
    if !isempty(del_fs_combined)
        @printf("  deletion_pad=%.1f: del_valid=%.3f +/- %.3f, del_combined=%.3f +/- %.3f (n_success=%d/%d)\n",
                dp, mean(del_fs_valid), std(del_fs_valid), mean(del_fs_combined), std(del_fs_combined), n_success, 10)
    else
        @printf("  deletion_pad=%.1f: ALL FAILED (negative Poisson lambda)\n", dp)
    end
end

# ============================================================================
# Deep dive: How deletion_pad creates deletions
# ============================================================================
println("\n\n" * "=" ^ 70)
println("DEEP DIVE: How deletion_pad works step by step")
println("=" ^ 70)

# Take one protein and trace through the deletion padding
Random.seed!(42)
idx = test_indices[1]
protein = proteins[idx]
x1_len = Int(sum(protein.mask .> 0.5))
println("\nSingle protein analysis (protein #$idx, length=$x1_len)")

x1 = protein_to_X1_simple(protein)
println("  X1 before padding: L=$(length(x1.groupings)), del=$(sum(x1.del)), flowmask=$(sum(x1.flowmask)), branchmask=$(sum(x1.branchmask))")

# Manually compute what deletion_pad does
gp = x1.groupings
X1_lengths_map = my_countmap(gp)
println("  Group lengths: $X1_lengths_map")

# resolved_mins from Poisson(X0_mean_length) - run multiple times
println("  resolved_mins samples (1 + Poisson($X0_mean_length)):")
for trial in 1:5
    resolved_mins = Dict(k => 1 + rand(Poisson(X0_mean_length)) for k in unique(gp))
    println("    trial $trial: $resolved_mins")
end

# One specific example
Random.seed!(42)
resolved_mins = Dict(k => 1 + rand(Poisson(X0_mean_length)) for k in unique(gp))
println("  Using resolved_mins: $resolved_mins")

# deletion_pad_counts
println("\n  deletion_pad=$deletion_pad computation:")
for j in keys(X1_lengths_map)
    total_exp = deletion_pad * max(X1_lengths_map[j], resolved_mins[j])
    lam = total_exp - X1_lengths_map[j]
    @printf("    group %d: total_expected = %.1f * max(%d, %d) = %.1f\n", j, deletion_pad, X1_lengths_map[j], resolved_mins[j], total_exp)
    @printf("             pad_count ~ Poisson(%.1f - %d) = Poisson(%.1f)\n", total_exp, X1_lengths_map[j], lam)
    if lam < 0
        println("             WARNING: negative Poisson lambda! This is a BUG.")
    else
        sampled = rand(Poisson(lam))
        println("             sampled: $sampled deletions to add")
    end
end

# ============================================================================
# Key insight: when X1_len > deletion_pad * resolved_min, the formula gives
# deletion_pad * X1_len - X1_len = (deletion_pad - 1) * X1_len
# So for deletion_pad=1.1, we add ~0.1 * X1_len deletions (about 10%)
# ============================================================================
println("\n  KEY FORMULA:")
println("  When X1_len >= X0_min (common for longer proteins):")
println("    pad_count ~ Poisson(deletion_pad * X1_len - X1_len)")
println("              = Poisson((deletion_pad - 1) * X1_len)")
println("              = Poisson($(deletion_pad - 1) * X1_len)")
println("    For X1_len=$x1_len: Poisson($(round((deletion_pad - 1) * x1_len, digits=1)))")
println("    Expected del fraction from padding alone: $(round((deletion_pad - 1) / deletion_pad, digits=4))")
println("")
println("  When X1_len < X0_min (short proteins relative to sampled X0):")
println("    pad_count ~ Poisson(deletion_pad * X0_min - X1_len)")
println("    This can be MUCH larger, causing HIGH deletion fractions!")
println("    Example: X1_len=70, X0_min=100:")
println("      pad_count ~ Poisson(1.1 * 100 - 70) = Poisson(40)")
println("      Expected del fraction: 40/110 = $(round(40/110, digits=3))")

# ============================================================================
# Also check: the forest_bridge may REMOVE positions via survival filtering
# at t (deletion_time_dist survival), which would reduce del further
# ============================================================================
println("\n\n" * "=" ^ 70)
println("DELETION SURVIVAL FILTERING IN tree_bridge")
println("=" ^ 70)
println("  In tree_bridge, deleted positions may be REMOVED before reaching Xt")
println("  if they fail to survive from current_t to target_t.")
println("  Survival probability = S(target_t) / S(current_t)")
println("  where S(t) = 1 - cdf(deletion_time_dist, t)")
println("  For Uniform(0,1) deletion_time_dist: S(t) = 1 - t")
println("  So survival from 0 to t is (1-t)/(1-0) = 1-t")
println("  At t=0.5: 50% of del=true positions are filtered out")
println("  At t=0.9: 10% survive (90% filtered)")
println("")
println("  This means at later times, fewer del=true positions appear in Xt,")
println("  but the ones that DO appear are genuine targets for the model.")

# ============================================================================
# Repeat analysis with more batches for statistics
# ============================================================================
println("\n\n" * "=" ^ 70)
println("STATISTICAL SUMMARY OVER 50 BATCHES")
println("=" ^ 70)

n_stat_batches = 50
del_fracs = Float64[]
del_frac_combined = Float64[]
xt_lengths_all = Int[]
x1_lengths_all = Int[]
t_values_all = Float64[]

for _ in 1:n_stat_batches
    idxs = rand(1:length(proteins), batch_size)
    X1s = [protein_to_X1_simple(proteins[i]) for i in idxs]

    batch = branching_bridge(
        P, X0_sampler_train, X1s, Uniform(0f0, 1f0);
        coalescence_factor = 1.0,
        use_branching_time_prob = 0.5,
        length_mins = Poisson(X0_mean_length),
        deletion_pad = deletion_pad
    )

    del = batch.del
    padmask = batch.Xt.padmask
    branchmask = batch.Xt.branchmask
    L_batch, B = size(del)

    for b in 1:B
        valid = padmask[:, b]
        combined = valid .& branchmask[:, b]
        n_valid = sum(valid)
        n_combined = sum(combined)
        n_del_valid = sum(del[:, b] .& valid)
        n_del_combined = sum(del[:, b] .& combined)

        push!(del_fracs, n_valid > 0 ? n_del_valid / n_valid : 0.0)
        push!(del_frac_combined, n_combined > 0 ? n_del_combined / n_combined : 0.0)
        push!(xt_lengths_all, n_valid)
        push!(x1_lengths_all, Int(sum(proteins[idxs[b]].mask .> 0.5)))
    end
    append!(t_values_all, batch.t)
end

println("\nOver $(n_stat_batches) batches ($(n_stat_batches * batch_size) samples):")
println("  Deletion fraction (among valid):")
println("    mean=$(round(mean(del_fracs), digits=4)), std=$(round(std(del_fracs), digits=4))")
println("    min=$(round(minimum(del_fracs), digits=4)), max=$(round(maximum(del_fracs), digits=4))")
println("    median=$(round(median(del_fracs), digits=4))")

println("  Deletion fraction (among combined=pad&branch, i.e. in loss):")
println("    mean=$(round(mean(del_frac_combined), digits=4)), std=$(round(std(del_frac_combined), digits=4))")
println("    min=$(round(minimum(del_frac_combined), digits=4)), max=$(round(maximum(del_frac_combined), digits=4))")
println("    median=$(round(median(del_frac_combined), digits=4))")

println("  Xt lengths: mean=$(round(mean(xt_lengths_all), digits=1)), std=$(round(std(xt_lengths_all), digits=1)), min=$(minimum(xt_lengths_all)), max=$(maximum(xt_lengths_all))")
println("  X1 lengths: mean=$(round(mean(x1_lengths_all), digits=1)), std=$(round(std(x1_lengths_all), digits=1)), min=$(minimum(x1_lengths_all)), max=$(maximum(x1_lengths_all))")
println("  Xt/X1 ratio: mean=$(round(mean(xt_lengths_all ./ x1_lengths_all), digits=3))")
println("  t values: mean=$(round(mean(t_values_all), digits=4)), std=$(round(std(t_values_all), digits=4)), min=$(round(minimum(t_values_all), digits=4)), max=$(round(maximum(t_values_all), digits=4))")

# Histogram of deletion fractions
println("\n  Deletion fraction histogram (among combined, in loss):")
bins = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for i in 1:length(bins)-1
    count = sum(bins[i] .<= del_frac_combined .< bins[i+1])
    pct = round(100 * count / length(del_frac_combined), digits=1)
    bar = repeat("#", Int(round(pct / 2)))
    @printf("    [%.2f, %.2f): %3d (%5.1f%%) %s\n", bins[i], bins[i+1], count, pct, bar)
end

# ============================================================================
# Analysis: correlation between t and deletion fraction
# ============================================================================
println("\n\n" * "=" ^ 70)
println("DELETION FRACTION vs TIME (t)")
println("=" ^ 70)

# Re-run with fixed times to see how del fraction varies with t
println("\nRunning batches with controlled t values...")
for t_val in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    del_fs = Float64[]
    for _ in 1:20
        idxs = rand(1:length(proteins), batch_size)
        X1s = [protein_to_X1_simple(proteins[i]) for i in idxs]

        # Use fixed times instead of sampling
        fixed_times = fill(Float32(t_val), batch_size)

        batch = branching_bridge(
            P, X0_sampler_train, X1s, fixed_times;
            coalescence_factor = 1.0,
            use_branching_time_prob = 0.0,  # Don't override with branching time
            length_mins = Poisson(X0_mean_length),
            deletion_pad = deletion_pad
        )

        del = batch.del
        padmask = batch.Xt.padmask
        branchmask = batch.Xt.branchmask
        L_batch, B = size(del)

        for b in 1:B
            combined = padmask[:, b] .& branchmask[:, b]
            n_combined = sum(combined)
            n_del_combined = sum(del[:, b] .& combined)
            push!(del_fs, n_combined > 0 ? n_del_combined / n_combined : 0.0)
        end
    end
    @printf("  t=%.1f: mean del fraction = %.4f (std=%.4f, n=%d)\n", t_val, mean(del_fs), std(del_fs), length(del_fs))
end

# ============================================================================
# CRITICAL: Check branchmask behavior
# ============================================================================
println("\n\n" * "=" ^ 70)
println("BRANCHMASK ANALYSIS")
println("=" ^ 70)
println("  In branching_bridge, branchmask is populated from batch_bridge[b][i].branchable")
println("  The default for ALL positions is branchmask=true (ones(Bool, maxlen, b))")
println("  After filling from the bridge data, padded positions also remain true!")
println("  This means combined_mask = padmask .* branchmask == padmask for padded positions.")
println("")
println("  BUT: branchmask is initialized to ones(Bool, maxlen, b) BEFORE the loop.")
println("  So positions beyond the actual data have branchmask=true but padmask=false.")
println("  The combined mask correctly excludes them.")
println("")
println("  However, the loss uses combined_mask (pad & branch). Since branchmask")  
println("  is always true for valid positions in these single-group proteins,")
println("  combined_mask == padmask for all valid positions.")

println("\n" * "=" ^ 70)
println("DIAGNOSTIC COMPLETE")
println("=" ^ 70)
