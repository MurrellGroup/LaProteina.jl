#!/usr/bin/env julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Plots
using DelimitedFiles
using Statistics

# Find the output directory from command line or most recent
if length(ARGS) >= 1
    output_dir = ARGS[1]
else
    outputs_root = joinpath(@__DIR__, "..", "outputs")
    dirs = filter(d -> startswith(d, "branching_full_"), readdir(outputs_root))
    sort!(dirs)
    output_dir = joinpath(outputs_root, dirs[end])
end

log_file = joinpath(output_dir, "training_log.txt")
println("Reading: $log_file")

# Parse CSV, skipping comment lines
lines = filter(l -> !startswith(l, "#") && length(l) > 0, readlines(log_file))
n = length(lines)
println("Found $n data points")

batch = zeros(Int, n)
total_loss = zeros(Float32, n)
ca_loss = zeros(Float32, n)
ll_loss = zeros(Float32, n)
split_loss = zeros(Float32, n)
del_loss = zeros(Float32, n)

for (i, line) in enumerate(lines)
    parts = split(line, ",")
    batch[i] = parse(Int, parts[1])
    # columns: batch,shard,lr,total_loss,ca_scaled,ll_scaled,split,del,t_min,t_max,time_ms
    total_loss[i] = parse(Float32, parts[4])
    ca_loss[i] = parse(Float32, parts[5])
    ll_loss[i] = parse(Float32, parts[6])
    split_loss[i] = parse(Float32, parts[7])
    del_loss[i] = parse(Float32, parts[8])
end

# Running average
function running_avg(x, window)
    out = similar(x)
    for i in eachindex(x)
        lo = max(1, i - window + 1)
        out[i] = mean(@view x[lo:i])
    end
    return out
end

w = 50

# Plot 1: Total loss
p1 = plot(batch, total_loss, alpha=0.15, color=:blue, label="raw", linewidth=0.5,
          xlabel="Batch", ylabel="Loss", title="Total Loss (scaled)")
plot!(p1, batch, running_avg(total_loss, w), color=:blue, linewidth=2, label="avg(50)")
savefig(p1, joinpath(output_dir, "samples", "loss_total.png"))
println("Saved loss_total.png")

# Plot 2: CA loss
p2 = plot(batch, ca_loss, alpha=0.15, color=:red, label="raw", linewidth=0.5,
          xlabel="Batch", ylabel="Loss", title="CA Loss (scaled, ×0.5)")
plot!(p2, batch, running_avg(ca_loss, w), color=:red, linewidth=2, label="avg(50)")
savefig(p2, joinpath(output_dir, "samples", "loss_ca.png"))
println("Saved loss_ca.png")

# Plot 3: LL loss
p3 = plot(batch, ll_loss, alpha=0.15, color=:green, label="raw", linewidth=0.5,
          xlabel="Batch", ylabel="Loss", title="LL Loss (scaled, ×0.1)")
plot!(p3, batch, running_avg(ll_loss, w), color=:green, linewidth=2, label="avg(50)")
savefig(p3, joinpath(output_dir, "samples", "loss_ll.png"))
println("Saved loss_ll.png")

# Plot 4: Split loss
p4 = plot(batch, split_loss, alpha=0.15, color=:orange, label="raw", linewidth=0.5,
          xlabel="Batch", ylabel="Loss", title="Split Loss")
plot!(p4, batch, running_avg(split_loss, w), color=:orange, linewidth=2, label="avg(50)")
savefig(p4, joinpath(output_dir, "samples", "loss_split.png"))
println("Saved loss_split.png")

# Plot 5: Del loss
p5 = plot(batch, del_loss, alpha=0.15, color=:purple, label="raw", linewidth=0.5,
          xlabel="Batch", ylabel="Loss", title="Del Loss")
plot!(p5, batch, running_avg(del_loss, w), color=:purple, linewidth=2, label="avg(50)")
savefig(p5, joinpath(output_dir, "samples", "loss_del.png"))
println("Saved loss_del.png")

# Plot 6: All components overlaid (running averages only)
p6 = plot(batch, running_avg(ca_loss, w), color=:red, linewidth=2, label="CA (×0.5)",
          xlabel="Batch", ylabel="Loss (scaled)", title="All Loss Components (avg 50)")
plot!(p6, batch, running_avg(ll_loss, w), color=:green, linewidth=2, label="LL (×0.1)")
plot!(p6, batch, running_avg(split_loss, w), color=:orange, linewidth=2, label="Split")
plot!(p6, batch, running_avg(del_loss, w), color=:purple, linewidth=2, label="Del")
savefig(p6, joinpath(output_dir, "samples", "loss_all_components.png"))
println("Saved loss_all_components.png")

println("\nAll plots saved to: $(joinpath(output_dir, "samples"))")
