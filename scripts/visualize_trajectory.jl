#!/usr/bin/env julia
# Visualize branching flow sampling trajectory as a two-panel video.
# Left panel: Xt (current noisy state), Right panel: X1hat (predicted clean structure).
# Last ~30% of trajectory: also shows backbone + sidechain ball-and-stick from decoder.
# After sampling frames, a "showcase" ending renders the final structure with ribbon + ball-and-stick.
#
# Usage:
#   xvfb-run julia --project=/home/claudey/JuProteina/run scripts/visualize_trajectory.jl <trajectory.jld2> [output.mp4]
#
# Does NOT load LaProteina/CUDA — only needs GLMakie, ProtPlot, JLD2.

using GLMakie
using ProtPlot
using JLD2
using Colors
using Statistics
using LinearAlgebra

# ── Parse arguments ──────────────────────────────────────────────────────

if isempty(ARGS)
    error("Usage: visualize_trajectory.jl <trajectory.jld2> [output.mp4]")
end
traj_path = ARGS[1]
output_path = length(ARGS) >= 2 ? ARGS[2] : replace(traj_path, ".jld2" => ".mp4")

println("Loading trajectory: $traj_path")
traj = load(traj_path)
frames = traj["frames"]
final_decoder = traj["final_decoder"]
metadata = traj["metadata"]

nframes = length(frames)
final_L = metadata["final_length"]
println("  $nframes frames, final length=$final_L")

# ── Subsampling — slower, denser ─────────────────────────────────────────

# Every 2nd frame for ~250 trajectory frames, 120 showcase frames, 20fps
frame_step = max(1, nframes ÷ 250)
sampled_indices = collect(1:frame_step:nframes)
if sampled_indices[end] != nframes
    push!(sampled_indices, nframes)
end
n_traj_frames = length(sampled_indices)
n_showcase_frames = 120  # 6s at 20fps
n_total_frames = n_traj_frames + n_showcase_frames
FPS = 20
println("  $n_traj_frames traj + $n_showcase_frames showcase = $n_total_frames total @ $(FPS)fps")

# ── Helpers ──────────────────────────────────────────────────────────────

nm_to_angstrom(x) = x .* 10f0

function center_coords(ca::AbstractMatrix)
    com = mean(ca, dims=2)
    return ca .- com
end

const BACKBONE_ATOM_IDS = Set([1, 2, 3, 5])  # N, CA, C, O in atom37
const BOND_CUTOFF = 1.85f0  # Angstroms

function residue_colors_rgb(L::Int)
    L <= 1 && return [RGB{Float32}(1f0, 0f0, 0f0)]
    [convert(RGB{Float32}, HSV(h, 0.85, 0.95)) for h in range(0, 300, length=L)]
end

# Build ball-and-stick data from all-atom coords
# include_backbone=true: draw all atoms + all intra-residue bonds (for trajectory frames)
# include_backbone=false: sidechain only, skip backbone-backbone bonds (for showcase with ribbon)
function build_ball_and_stick(all_atom_c, atom_mask, L, colors; include_backbone::Bool=false)
    sc_pos = Point3f[]
    sc_col = RGB{Float32}[]
    bond_seg = Point3f[]
    bond_col = RGB{Float32}[]

    for res in 1:L
        col = colors[res]
        aidx = Int[]
        apos = Point3f[]
        for j in 1:37
            if atom_mask[j, res]
                push!(aidx, j)
                push!(apos, Point3f(all_atom_c[:, j, res]))
            end
        end

        # Atom spheres
        for (k, j) in enumerate(aidx)
            if include_backbone || j ∉ BACKBONE_ATOM_IDS
                push!(sc_pos, apos[k])
                push!(sc_col, col)
            end
        end

        # Bonds (distance-based, intra-residue only)
        for ia in 1:length(aidx)
            for ib in (ia+1):length(aidx)
                a, b = aidx[ia], aidx[ib]
                if !include_backbone
                    a ∈ BACKBONE_ATOM_IDS && b ∈ BACKBONE_ATOM_IDS && continue
                end
                d = norm(apos[ia] - apos[ib])
                if 0.5f0 < d < BOND_CUTOFF
                    push!(bond_seg, apos[ia])
                    push!(bond_seg, apos[ib])
                    push!(bond_col, col)
                    push!(bond_col, col)
                end
            end
        end
    end
    return sc_pos, sc_col, bond_seg, bond_col
end

# ── Prepare final decoder data (for showcase) ───────────────────────────

all_atom = nm_to_angstrom(Float32.(final_decoder[:all_atom_coords]))  # [3, 37, L]
atom_mask = Bool.(final_decoder[:atom_mask])                           # [37, L]

# Backbone for ribbon: [3, 3, L] = [xyz, (N,CA,C), residue]
backbone = zeros(Float32, 3, 3, final_L)
backbone[:, 1, :] = all_atom[:, 1, :]  # N
backbone[:, 2, :] = all_atom[:, 2, :]  # CA
backbone[:, 3, :] = all_atom[:, 3, :]  # C

# Center on CA center-of-mass
bb_com = mean(backbone[:, 2, :], dims=2)
backbone = backbone .- bb_com

# Centered all-atom
all_atom_c = similar(all_atom)
for j in 1:37
    all_atom_c[:, j, :] = all_atom[:, j, :] .- bb_com
end

# Showcase ball-and-stick
colors_final = residue_colors_rgb(final_L)
sc_positions, sc_colors, bond_segments, bond_seg_colors = build_ball_and_stick(
    all_atom_c, atom_mask, final_L, colors_final)

# Ribbon colors: explicit per-residue to match our rainbow scheme
ribbon_colors = [Float64(i-1) / max(final_L-1, 1) for i in 1:final_L]

println("  Showcase: $(length(sc_positions)) sidechain atoms, $(length(bond_segments)÷2) bonds")

# ── Scene bounds and camera ──────────────────────────────────────────────

final_ca_c = center_coords(nm_to_angstrom(Float32.(final_decoder[:ca_coords])))
max_extent = maximum(abs.(final_ca_c))
# Single camera radius — no jump at transition
cam_radius = max(max_extent * 2.2, 25f0)

# Custom colormap matching our residue_colors_rgb
rainbow_cmap = cgrad([residue_colors_rgb(256)[i] for i in 1:256])

# ── Build figure ─────────────────────────────────────────────────────────

GLMakie.activate!(; visible=false)
fig = Figure(size=(2400, 800), backgroundcolor=RGBf(0.97, 0.97, 0.97))

ax_left = LScene(fig[1, 1]; show_axis=false)
Box(fig[1, 2]; color=RGBf(0.3, 0.3, 0.3), width=3)
ax_right = LScene(fig[1, 3]; show_axis=false)

colsize!(fig.layout, 1, Relative(0.498))
colsize!(fig.layout, 2, Fixed(3))
colsize!(fig.layout, 3, Relative(0.498))

Label(fig[0, 1], "Xₜ  (current state)", fontsize=26, halign=:center, font=:bold)
Label(fig[0, 3], "X̂₁  (model prediction)", fontsize=26, halign=:center, font=:bold)
time_label = Label(fig[2, 1:3], "", fontsize=20, halign=:center)

# ── Helper: set camera on an LScene ──────────────────────────────────────

function set_camera!(ax, eyepos, lookat=Vec3f(0), up=Vec3f(0,0,1); fov=50f0)
    cam3d!(ax.scene; center=false)
    cam = cameracontrols(ax)
    cam.eyeposition[] = eyepos
    cam.lookat[] = lookat
    cam.upvector[] = up
    cam.fov[] = fov
end

# ── Helper: render all-atom overlay (ball-and-stick) for a frame ─────────

function render_allatom_overlay!(ax, all_atom_raw, amask, L, com_offset; alpha=0.7f0)
    # Convert and center
    aa = nm_to_angstrom(Float32.(all_atom_raw))  # [3, 37, L]
    aa_c = similar(aa)
    for j in 1:37
        aa_c[:, j, :] = aa[:, j, :] .- com_offset
    end

    cols = residue_colors_rgb(L)
    sc_pos, sc_col, bseg, bcol = build_ball_and_stick(aa_c, amask, L, cols; include_backbone=true)

    if !isempty(sc_pos)
        sc_col_a = [RGBA{Float32}(c.r, c.g, c.b, alpha) for c in sc_col]
        meshscatter!(ax, sc_pos; color=sc_col_a, markersize=0.15, transparency=true)
    end
    if !isempty(bseg)
        bcol_a = [RGBA{Float32}(c.r, c.g, c.b, alpha) for c in bcol]
        linesegments!(ax, bseg; color=bcol_a, linewidth=5)
    end
end

# ── Record video ─────────────────────────────────────────────────────────

println("Recording to: $output_path")

record(fig, output_path, 1:n_total_frames; framerate=FPS) do frame_idx
    empty!(ax_left.scene)
    empty!(ax_right.scene)

    # Camera orbit — consistent radius and elevation throughout
    azimuth = 2π * frame_idx / n_total_frames
    elevation = π / 7
    eyepos = Vec3f(
        cam_radius * sin(azimuth) * cos(elevation),
        cam_radius * cos(azimuth) * cos(elevation),
        cam_radius * sin(elevation)
    )

    if frame_idx <= n_traj_frames
        # ── Trajectory frame ──
        si = sampled_indices[frame_idx]
        f = frames[si]
        t_val = f[:t]
        step = f[:step]
        L = f[:L]

        xt_ca = center_coords(nm_to_angstrom(Float32.(f[:xt_ca])))
        x1hat_ca = center_coords(nm_to_angstrom(Float32.(f[:x1hat_ca])))

        cols = residue_colors_rgb(L)
        cols_rgba = [RGBA{Float32}(c.r, c.g, c.b, 0.9) for c in cols]
        sphere_r = L < 20 ? 0.6f0 : L < 60 ? 0.4f0 : 0.27f0

        if L > 0
            pts_xt = [Point3f(xt_ca[:, i]) for i in 1:L]
            pts_x1 = [Point3f(x1hat_ca[:, i]) for i in 1:L]
            meshscatter!(ax_left, pts_xt; color=cols_rgba, markersize=sphere_r)
            meshscatter!(ax_right, pts_x1; color=cols_rgba, markersize=sphere_r)
        end

        # Show sidechain ball-and-stick for entire trajectory
        if L > 0
            xt_com = mean(nm_to_angstrom(Float32.(f[:xt_ca])), dims=2)
            x1_com = mean(nm_to_angstrom(Float32.(f[:x1hat_ca])), dims=2)

            if haskey(f, :x1hat_all_atom)
                render_allatom_overlay!(ax_right, f[:x1hat_all_atom], Bool.(f[:x1hat_atom_mask]), L, x1_com)
            end
            if haskey(f, :xt_all_atom)
                render_allatom_overlay!(ax_left, f[:xt_all_atom], Bool.(f[:xt_atom_mask]), L, xt_com)
            end
        end

        time_label.text[] = "Step $(step)/$(metadata["nsteps"])  |  t = $(round(t_val, digits=3))  |  L = $L"
    else
        # ── Showcase frame: ribbon + ball-and-stick ──
        ribbon!(ax_left, backbone; colors=[ribbon_colors], colormap=rainbow_cmap)
        ribbon!(ax_right, backbone; colors=[ribbon_colors], colormap=rainbow_cmap)

        if !isempty(sc_positions)
            meshscatter!(ax_left, sc_positions; color=sc_colors, markersize=0.15)
            meshscatter!(ax_right, sc_positions; color=sc_colors, markersize=0.15)
        end
        if !isempty(bond_segments)
            linesegments!(ax_left, bond_segments; color=bond_seg_colors, linewidth=5)
            linesegments!(ax_right, bond_segments; color=bond_seg_colors, linewidth=5)
        end

        time_label.text[] = "Final structure  |  L = $final_L"
    end

    # Set camera after plots
    set_camera!(ax_left, eyepos)
    set_camera!(ax_right, eyepos)
end

println("Done! Video saved to: $output_path")
