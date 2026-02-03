# PDB file loading utilities
# Uses BioStructures.jl for parsing

import BioStructures: read, PDBFormat, MMCIFFormat, collectresidues, standardselector,
                      collectatoms, resname, atomname
import BioStructures: coords as bio_coords

"""
    load_pdb(filepath::String; chain_id::String="A")

Load a PDB or mmCIF file and extract protein structure data.

# Arguments
- `filepath`: Path to PDB or CIF file (format auto-detected from extension)
- `chain_id`: Chain ID to extract (default "A")

# Returns
Dict with:
- :coords => [3, 37, L] atom coordinates in nanometers
- :aatype => [L] amino acid indices (1-20)
- :atom_mask => [37, L] boolean mask of present atoms
- :residue_mask => [L] boolean mask (all true)
- :sequence => amino acid sequence string
"""
function load_pdb(filepath::String; chain_id::String="A")
    # Auto-detect format from extension
    ext = lowercase(splitext(filepath)[2])
    if ext in [".cif", ".mmcif"]
        struc = read(filepath, MMCIFFormat)
    else
        struc = read(filepath, PDBFormat)
    end

    # Get chain
    chain = struc[chain_id]
    residues = collectresidues(chain, standardselector)

    L = length(residues)

    # Initialize arrays
    coords = zeros(Float32, 3, ATOM_TYPE_NUM, L)
    atom_mask = zeros(Bool, ATOM_TYPE_NUM, L)
    aatype = zeros(Int, L)
    sequence = Vector{Char}(undef, L)

    for (i, res) in enumerate(residues)
        # Get residue name and convert to index
        res_name = resname(res)
        aa_char = get(RESTYPE_3TO1, res_name, 'X')
        aatype[i] = aa_to_index(aa_char)
        sequence[i] = aa_char

        # Get atom coordinates
        for atom in collectatoms(res)
            atom_nm = atomname(atom)

            # Check if atom is in our atom37 representation
            if haskey(ATOM_ORDER, atom_nm)
                atom_idx = ATOM_ORDER[atom_nm]
                coord = bio_coords(atom)  # Returns 3-element vector in Angstroms

                # Convert to nanometers
                coords[:, atom_idx, i] = Float32.(coord) ./ 10.0f0
                atom_mask[atom_idx, i] = true
            end
        end
    end

    residue_mask = ones(Bool, L)

    return Dict(
        :coords => coords,           # [3, 37, L]
        :aatype => aatype,           # [L]
        :atom_mask => atom_mask,     # [37, L]
        :residue_mask => residue_mask, # [L]
        :sequence => String(sequence)
    )
end

"""
    extract_ca_coords(data::Dict)

Extract CA coordinates from loaded PDB data.

# Returns
CA coordinates [3, L] in nanometers
"""
function extract_ca_coords(data::Dict)
    return data[:coords][:, CA_INDEX, :]  # [3, L]
end

"""
    batch_pdb_data(data_list::Vector{Dict}; pad_length::Union{Int,Nothing}=nothing)

Batch multiple PDB data dictionaries, padding to uniform length.

# Arguments
- `data_list`: Vector of dicts from load_pdb
- `pad_length`: Length to pad to (default: max length in batch)

# Returns
Dict with batched arrays (dimension order: [feature_dims..., L, B])
"""
function batch_pdb_data(data_list::Vector{<:Dict}; pad_length::Union{Int,Nothing}=nothing)
    B = length(data_list)

    # Determine padding length
    lengths = [size(d[:coords], 3) for d in data_list]
    L = isnothing(pad_length) ? maximum(lengths) : pad_length

    # Initialize batched arrays
    coords = zeros(Float32, 3, ATOM_TYPE_NUM, L, B)
    aatype = zeros(Int, L, B)
    atom_mask = zeros(Bool, ATOM_TYPE_NUM, L, B)
    residue_mask = zeros(Bool, L, B)

    for (b, data) in enumerate(data_list)
        len_b = size(data[:coords], 3)
        coords[:, :, 1:len_b, b] = data[:coords]
        aatype[1:len_b, b] = data[:aatype]
        atom_mask[:, 1:len_b, b] = data[:atom_mask]
        residue_mask[1:len_b, b] .= true
    end

    return Dict(
        :coords => coords,           # [3, 37, L, B]
        :aatype => aatype,           # [L, B]
        :atom_mask => atom_mask,     # [37, L, B]
        :residue_mask => residue_mask, # [L, B]
        :mask => residue_mask        # Alias
    )
end

"""
    save_pdb(filepath::String, coords::AbstractArray{<:Real,3},
             aatype::AbstractVector{<:Integer};
             atom_mask::Union{AbstractMatrix{Bool},Nothing}=nothing,
             chain_id::String="A")

Save structure to PDB file.

# Arguments
- `filepath`: Output file path
- `coords`: Atom coordinates [3, 37, L] in nanometers
- `aatype`: Amino acid indices [L]
- `atom_mask`: Optional atom mask [37, L]
- `chain_id`: Chain identifier
"""
function save_pdb(filepath::String, coords::AbstractArray{<:Real,3},
                  aatype::AbstractVector{<:Integer};
                  atom_mask::Union{AbstractMatrix{Bool},Nothing}=nothing,
                  chain_id::String="A")

    L = size(coords, 3)

    if isnothing(atom_mask)
        # Generate from aatype
        atom_mask = zeros(Bool, ATOM_TYPE_NUM, L)
        for i in 1:L
            aa = clamp(aatype[i], 1, 21)
            atom_mask[:, i] = RESTYPE_ATOM37_MASK[aa, :]
        end
    end

    open(filepath, "w") do io
        atom_serial = 0

        for res_idx in 1:L
            aa_idx = clamp(aatype[res_idx], 1, 20)
            aa_char = index_to_aa(aa_idx)
            resname = RESTYPE_1TO3[aa_char]

            for (atom_idx, atom_name) in enumerate(ATOM_TYPES)
                if atom_mask[atom_idx, res_idx]
                    atom_serial += 1
                    # Convert nm to Angstroms
                    x = coords[1, atom_idx, res_idx] * 10
                    y = coords[2, atom_idx, res_idx] * 10
                    z = coords[3, atom_idx, res_idx] * 10

                    # PDB ATOM record format
                    element = atom_name[1:1]
                    @printf(io, "ATOM  %5d %-4s %3s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n",
                            atom_serial, atom_name, resname, chain_id, res_idx, x, y, z, element)
                end
            end
        end

        println(io, "END")
    end
end

# Helper for Printf
using Printf
