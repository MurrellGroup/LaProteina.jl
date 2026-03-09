# Motif structure extraction from PDB files
# Port of extract_motif_from_pdb from proteinfoundation/utils/motif_utils.py

import BioStructures: read, PDBFormat, MMCIFFormat, collectresidues, standardselector,
                      collectatoms, resname, atomname
import BioStructures: coords as bio_coords

"""
    extract_motif_from_pdb(pdb_path::String, segments::Vector{<:ContigSegment};
                            atom_selection::Symbol=:all_atom)

Extract motif atom coordinates and masks from a PDB file.

# Arguments
- `pdb_path`: Path to PDB file
- `segments`: Parsed contig segments (from `parse_contig`)
- `atom_selection`: One of `:ca`, `:backbone`, `:all_atom`, `:tip_atoms`

# Returns
Named tuple with:
- `motif_mask`: [37, L_motif] Bool mask of present atoms
- `x_motif`: [3, 37, L_motif] Float32 coordinates in nanometers
- `residue_types`: [L_motif] Int residue type indices (1-20)
- `chain_residues`: Vector of (chain, start, end) tuples for motif segments
"""
function extract_motif_from_pdb(pdb_path::String, segments::Vector{<:ContigSegment};
                                 atom_selection::Symbol=:all_atom)
    @assert atom_selection in (:ca, :backbone, :all_atom, :tip_atoms) "Invalid atom_selection: $atom_selection"

    # Collect motif segments
    motif_segs = [(s.chain, s.start_res, s.end_res) for s in segments if s isa MotifSegment]
    isempty(motif_segs) && error("No motif segments found in contig")

    # Load PDB
    ext = lowercase(splitext(pdb_path)[2])
    struc = if ext in [".cif", ".mmcif"]
        read(pdb_path, MMCIFFormat)
    else
        read(pdb_path, PDBFormat)
    end

    # Extract motif residues
    total_motif_len = sum(e - s + 1 for (_, s, e) in motif_segs)
    motif_mask = zeros(Bool, ATOM_TYPE_NUM, total_motif_len)
    x_motif = zeros(Float32, 3, ATOM_TYPE_NUM, total_motif_len)
    residue_types = zeros(Int, total_motif_len)

    pos = 0
    for (chain_id, start_res, end_res) in motif_segs
        chain = struc[string(chain_id)]
        residues = collectresidues(chain, standardselector)

        # Build residue index map
        res_map = Dict{Int, eltype(residues)}()
        for res in residues
            # Get residue number
            res_num = BioStructures.resnumber(res)
            res_map[res_num] = res
        end

        for res_idx in start_res:end_res
            pos += 1
            if !haskey(res_map, res_idx)
                continue
            end
            res = res_map[res_idx]

            # Get residue type
            res_name = resname(res)
            aa_char = get(RESTYPE_3TO1, res_name, 'X')
            residue_types[pos] = aa_to_index(aa_char)

            # Get atoms
            atoms = collectatoms(res)
            for atom in atoms
                aname = atomname(atom)
                aname = strip(aname)

                if !haskey(ATOM_ORDER, aname)
                    continue
                end
                atom_idx = ATOM_ORDER[aname]

                # Apply atom selection filter
                if !_should_include_atom(atom_idx, aname, res_name, atom_selection)
                    continue
                end

                # Store coordinates (convert Å to nm)
                c = bio_coords(atom)
                x_motif[1, atom_idx, pos] = Float32(c[1]) / 10.0f0
                x_motif[2, atom_idx, pos] = Float32(c[2]) / 10.0f0
                x_motif[3, atom_idx, pos] = Float32(c[3]) / 10.0f0
                motif_mask[atom_idx, pos] = true
            end
        end
    end

    return (motif_mask=motif_mask, x_motif=x_motif,
            residue_types=residue_types, chain_residues=motif_segs)
end

"""
    _should_include_atom(atom_idx, atom_name, res_name, selection) -> Bool

Check if an atom should be included based on atom selection mode.
"""
function _should_include_atom(atom_idx::Int, atom_name::String, res_name::String,
                               selection::Symbol)
    if selection == :all_atom
        return true
    elseif selection == :ca
        return atom_name == "CA"
    elseif selection == :backbone
        return atom_name in ("N", "CA", "C", "O")
    elseif selection == :tip_atoms
        # Backbone atoms always included
        if atom_name in ("N", "CA", "C", "O")
            return true
        end
        # Check sidechain tip atoms
        if haskey(SIDECHAIN_TIP_ATOMS, res_name)
            return atom_name in SIDECHAIN_TIP_ATOMS[res_name]
        end
        return false
    end
    return false
end
