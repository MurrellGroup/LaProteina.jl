# Atom37 representation and amino acid constants from OpenFold
# Julia port of openfold/np/residue_constants.py

# Distance from one CA to next CA [trans configuration: omega = 180]
const CA_CA_DISTANCE = 3.80209737096

# Atom types in OpenFold atom37 representation order
const ATOM_TYPES = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1",
    "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2",
    "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT"
]

# Atom order (1-indexed for Julia)
const ATOM_ORDER = Dict(atom => i for (i, atom) in enumerate(ATOM_TYPES))
const ATOM_TYPE_NUM = length(ATOM_TYPES)  # 37

# 20 standard amino acids in alphabetical order (1-letter codes)
const RESTYPES = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                  'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

const RESTYPE_ORDER = Dict(aa => i for (i, aa) in enumerate(RESTYPES))
const RESTYPE_NUM = length(RESTYPES)  # 20
const UNK_RESTYPE_INDEX = RESTYPE_NUM + 1  # 21

# 1-letter to 3-letter amino acid code mapping
const RESTYPE_1TO3 = Dict(
    'A' => "ALA", 'R' => "ARG", 'N' => "ASN", 'D' => "ASP", 'C' => "CYS",
    'Q' => "GLN", 'E' => "GLU", 'G' => "GLY", 'H' => "HIS", 'I' => "ILE",
    'L' => "LEU", 'K' => "LYS", 'M' => "MET", 'F' => "PHE", 'P' => "PRO",
    'S' => "SER", 'T' => "THR", 'W' => "TRP", 'Y' => "TYR", 'V' => "VAL"
)

const RESTYPE_3TO1 = Dict(v => k for (k, v) in RESTYPE_1TO3)

# Atoms present in each residue type (excluding hydrogen)
const RESIDUE_ATOMS = Dict{String, Vector{String}}(
    "ALA" => ["C", "CA", "CB", "N", "O"],
    "ARG" => ["C", "CA", "CB", "CG", "CD", "CZ", "N", "NE", "O", "NH1", "NH2"],
    "ASN" => ["C", "CA", "CB", "CG", "N", "ND2", "O", "OD1"],
    "ASP" => ["C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"],
    "CYS" => ["C", "CA", "CB", "N", "O", "SG"],
    "GLN" => ["C", "CA", "CB", "CG", "CD", "N", "NE2", "O", "OE1"],
    "GLU" => ["C", "CA", "CB", "CG", "CD", "N", "O", "OE1", "OE2"],
    "GLY" => ["C", "CA", "N", "O"],
    "HIS" => ["C", "CA", "CB", "CG", "CD2", "CE1", "N", "ND1", "NE2", "O"],
    "ILE" => ["C", "CA", "CB", "CG1", "CG2", "CD1", "N", "O"],
    "LEU" => ["C", "CA", "CB", "CG", "CD1", "CD2", "N", "O"],
    "LYS" => ["C", "CA", "CB", "CG", "CD", "CE", "N", "NZ", "O"],
    "MET" => ["C", "CA", "CB", "CG", "CE", "N", "O", "SD"],
    "PHE" => ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O"],
    "PRO" => ["C", "CA", "CB", "CG", "CD", "N", "O"],
    "SER" => ["C", "CA", "CB", "N", "O", "OG"],
    "THR" => ["C", "CA", "CB", "CG2", "N", "O", "OG1"],
    "TRP" => ["C", "CA", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "N", "NE1", "O"],
    "TYR" => ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O", "OH"],
    "VAL" => ["C", "CA", "CB", "CG1", "CG2", "N", "O"],
)

"""
    make_restype_atom37_mask()

Create a mask of shape (21, 37) indicating which atoms are present for each residue type.
Row i corresponds to residue type i (1-20 for standard AAs, 21 for unknown).
Column j corresponds to atom type j in ATOM_TYPES.
"""
function make_restype_atom37_mask()
    mask = zeros(Bool, RESTYPE_NUM + 1, ATOM_TYPE_NUM)
    for (restype_idx, restype_letter) in enumerate(RESTYPES)
        restype_name = RESTYPE_1TO3[restype_letter]
        atom_names = RESIDUE_ATOMS[restype_name]
        for atom_name in atom_names
            atom_idx = ATOM_ORDER[atom_name]
            mask[restype_idx, atom_idx] = true
        end
    end
    # Row 21 (unknown) is all zeros
    return mask
end

const RESTYPE_ATOM37_MASK = make_restype_atom37_mask()

# AA character classification (Aliphatic, Polar, Charged)
const AA_CHARACTER_PROTORP = Dict{String, Char}(
    "ALA" => 'A', "CYS" => 'P', "GLU" => 'C', "ASP" => 'C', "GLY" => 'A',
    "PHE" => 'A', "ILE" => 'A', "HIS" => 'P', "LYS" => 'C', "MET" => 'A',
    "LEU" => 'A', "ASN" => 'P', "GLN" => 'P', "PRO" => 'A', "SER" => 'P',
    "ARG" => 'C', "THR" => 'P', "TRP" => 'P', "VAL" => 'A', "TYR" => 'P',
)

# Sidechain tip atoms for each residue
const SIDECHAIN_TIP_ATOMS = Dict{String, Vector{String}}(
    "ALA" => ["CA", "CB"],
    "ARG" => ["CD", "CZ", "NE", "NH1", "NH2"],
    "ASP" => ["CB", "CG", "OD1", "OD2"],
    "ASN" => ["CB", "CG", "ND2", "OD1"],
    "CYS" => ["CA", "CB", "SG"],
    "GLU" => ["CG", "CD", "OE1", "OE2"],
    "GLN" => ["CG", "CD", "NE2", "OE1"],
    "GLY" => String[],
    "HIS" => ["CB", "CG", "CD2", "CE1", "ND1", "NE2"],
    "ILE" => ["CB", "CG1", "CG2", "CD1"],
    "LEU" => ["CB", "CG", "CD1", "CD2"],
    "LYS" => ["CE", "NZ"],
    "MET" => ["CG", "CE", "SD"],
    "PHE" => ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO" => ["CA", "CB", "CG", "CD", "N"],
    "SER" => ["CA", "CB", "OG"],
    "THR" => ["CA", "CB", "CG2", "OG1"],
    "TRP" => ["CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "NE1"],
    "TYR" => ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL" => ["CB", "CG1", "CG2"],
)

# CA atom index (1-indexed)
const CA_INDEX = ATOM_ORDER["CA"]  # Should be 2

"""
    get_atom_mask(aatype::AbstractArray{<:Integer})

Get the atom37 mask for a batch of residue types.

# Arguments
- `aatype`: Array of residue type indices (1-20 for standard AAs, 21 for unknown)

# Returns
- Boolean mask array with one more dimension than input
"""
function get_atom_mask(aatype::AbstractArray{<:Integer})
    return RESTYPE_ATOM37_MASK[aatype, :]
end

# Convenience functions for index conversion
"""
    aa_to_index(aa::Char) -> Int

Convert single-letter amino acid code to index (1-20).
Returns 21 for unknown amino acids.
"""
function aa_to_index(aa::Char)
    return get(RESTYPE_ORDER, aa, UNK_RESTYPE_INDEX)
end

"""
    index_to_aa(idx::Integer) -> Char

Convert index (1-20) to single-letter amino acid code.
Returns 'X' for index 21 or out of range.
"""
function index_to_aa(idx::Integer)
    if 1 <= idx <= RESTYPE_NUM
        return RESTYPES[idx]
    else
        return 'X'
    end
end

"""
    sequence_to_indices(seq::AbstractString) -> Vector{Int}

Convert amino acid sequence to vector of indices.
"""
function sequence_to_indices(seq::AbstractString)
    return [aa_to_index(aa) for aa in seq]
end

"""
    indices_to_sequence(indices::AbstractVector{<:Integer}) -> String

Convert vector of indices to amino acid sequence.
"""
function indices_to_sequence(indices::AbstractVector{<:Integer})
    return String([index_to_aa(i) for i in indices])
end
