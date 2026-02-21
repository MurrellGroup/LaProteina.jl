# Inference utilities
# get_schedule: time schedules for sampling (used by flowfusion_sampling.jl)
# samples_to_pdb: save generated samples as PDB files

"""
    get_schedule(mode::Symbol, nsteps::Int; p::Real=1.0)

Get time schedule for sampling steps.
Returns array of length nsteps+1 from 0 to 1.

# Arguments
- `mode`: :uniform, :power, or :log
- `nsteps`: Number of sampling steps
- `p`: Parameter for power/log schedules
"""
function get_schedule(mode::Symbol, nsteps::Int; p::Real=1.0)
    if mode == :uniform
        return Float32.(range(0, 1, length=nsteps+1))
    elseif mode == :power
        t = Float32.(range(0, 1, length=nsteps+1))
        return t .^ Float32(p)
    elseif mode == :log
        t = 1.0f0 .- Float32.(exp10.(range(-Float32(p), 0, length=nsteps+1)))
        t = reverse(t)
        t = t .- minimum(t)
        t = t ./ maximum(t)
        return t
    else
        error("Unknown schedule mode: $mode")
    end
end

"""
    samples_to_pdb(samples::Dict, output_dir::String;
                   prefix::String="sample",
                   save_all_atom::Bool=true)

Save generated samples as PDB files.

# Arguments
- `samples`: Dict with :ca_coords, :all_atom_coords, :aatype, :atom_mask keys
- `output_dir`: Directory to save PDB files
- `prefix`: Filename prefix
- `save_all_atom`: If true, save all-atom coords; if false, save CA only
"""
function samples_to_pdb(samples::Dict, output_dir::String;
                        prefix::String="sample",
                        save_all_atom::Bool=true)

    mkpath(output_dir)

    B = size(samples[:ca_coords], 3)

    for b in 1:B
        filename = joinpath(output_dir, "$(prefix)_$(b).pdb")

        if save_all_atom && haskey(samples, :all_atom_coords)
            coords = samples[:all_atom_coords][:, :, :, b]  # [3, 37, L]
            aatype = samples[:aatype][:, b]                  # [L]
            atom_mask = samples[:atom_mask][:, :, b]         # [37, L]
            save_pdb(filename, coords, aatype; atom_mask=atom_mask)
        else
            # CA-only PDB (simplified)
            ca_coords = samples[:ca_coords][:, :, b]  # [3, L]
            L = size(ca_coords, 2)

            # Create dummy all-atom (just CA)
            coords = zeros(Float32, 3, 37, L)
            coords[:, CA_INDEX, :] = ca_coords
            aatype = haskey(samples, :aatype) ? samples[:aatype][:, b] : fill(1, L)
            atom_mask = zeros(Bool, 37, L)
            atom_mask[CA_INDEX, :] .= true

            save_pdb(filename, coords, aatype; atom_mask=atom_mask)
        end

        @info "Saved $(filename)"
    end
end
