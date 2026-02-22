# Motif batch preparation for indexed motif scaffolding
# Port of pad_motif_to_full_length from proteinfoundation/utils/motif_utils.py

"""
    prepare_motif_batch(pdb_path::String, contig_string::String, n_samples::Int;
                         atom_selection::Symbol=:all_atom,
                         min_length::Int=50, max_length::Int=512)

Prepare a batch of motif-conditioned inputs for sampling.

# Arguments
- `pdb_path`: Path to PDB file containing motif structure
- `contig_string`: Contig specification (e.g., "10-40/A163-181/10-40")
- `n_samples`: Number of samples to generate
- `atom_selection`: One of `:ca`, `:backbone`, `:all_atom`, `:tip_atoms`
- `min_length`: Minimum total protein length
- `max_length`: Maximum total protein length

# Returns
Dict with batch tensors compatible with ScoreNetwork:
- `:motif_mask` => [37, L, B] Bool atom mask
- `:x_motif` => [3, 37, L, B] Float32 motif coordinates (centered)
- `:seq_motif` => [L, B] Int residue types (0 for scaffold)
- `:seq_motif_mask` => [L, B] Bool residue-level mask
- `:mask` => [L, B] Float32 sequence mask (all ones)
- `:motif_indices` => Vector{Vector{Int}} per-sample motif positions
- `:total_lengths` => Vector{Int} per-sample total lengths
"""
function prepare_motif_batch(pdb_path::String, contig_string::String, n_samples::Int;
                              atom_selection::Symbol=:all_atom,
                              min_length::Int=50, max_length::Int=512)
    # Parse contig
    segments = parse_contig(contig_string)

    # Extract motif from PDB
    motif_data = extract_motif_from_pdb(pdb_path, segments; atom_selection=atom_selection)

    # Generate scaffold lengths for each sample
    lengths_per_sample = generate_scaffold_lengths(segments;
        min_length=min_length, max_length=max_length, n_samples=n_samples)

    # Find max total length for padding
    total_lengths = [sum(l) for l in lengths_per_sample]
    L_max = maximum(total_lengths)

    # Allocate batch tensors
    motif_mask_batch = zeros(Bool, ATOM_TYPE_NUM, L_max, n_samples)
    x_motif_batch = zeros(Float32, 3, ATOM_TYPE_NUM, L_max, n_samples)
    seq_motif_batch = zeros(Int, L_max, n_samples)
    seq_motif_mask_batch = zeros(Bool, L_max, n_samples)
    mask_batch = zeros(Float32, L_max, n_samples)
    all_motif_indices = Vector{Vector{Int}}()

    for s in 1:n_samples
        lengths = lengths_per_sample[s]
        total_len = total_lengths[s]

        # Compute motif positions in full-length protein
        motif_indices = compute_motif_indices(segments, lengths)
        push!(all_motif_indices, motif_indices)

        # Set sequence mask for valid positions
        mask_batch[1:total_len, s] .= 1.0f0

        # Place motif at correct positions
        motif_pos = 0
        for idx in motif_indices
            motif_pos += 1
            if motif_pos > size(motif_data.motif_mask, 2)
                break
            end

            motif_mask_batch[:, idx, s] .= motif_data.motif_mask[:, motif_pos]
            x_motif_batch[:, :, idx, s] .= motif_data.x_motif[:, :, motif_pos]
            seq_motif_batch[idx, s] = motif_data.residue_types[motif_pos]
            seq_motif_mask_batch[idx, s] = true
        end
    end

    # Center motif coordinates to origin (per sample, using motif atoms only)
    for s in 1:n_samples
        _center_motif_coords!(view(x_motif_batch, :, :, :, s),
                               view(motif_mask_batch, :, :, s))
    end

    return Dict{Symbol, Any}(
        :motif_mask => motif_mask_batch,           # [37, L, B]
        :x_motif => x_motif_batch,                  # [3, 37, L, B]
        :seq_motif => seq_motif_batch,              # [L, B]
        :seq_motif_mask => seq_motif_mask_batch,    # [L, B]
        :mask => mask_batch,                         # [L, B]
        :motif_indices => all_motif_indices,
        :total_lengths => total_lengths,
    )
end

"""
    _center_motif_coords!(x_motif, motif_mask)

Center motif coordinates to have zero center of mass.
`x_motif`: [3, 37, L] coordinates, `motif_mask`: [37, L] mask.
Modifies x_motif in-place.
"""
function _center_motif_coords!(x_motif::AbstractArray{Float32, 3},
                                motif_mask::AbstractArray{Bool, 2})
    # Compute center of mass of motif atoms
    total_atoms = 0
    com = zeros(Float32, 3)

    n_atoms, L = size(motif_mask)
    for l in 1:L
        for a in 1:n_atoms
            if motif_mask[a, l]
                com .+= x_motif[:, a, l]
                total_atoms += 1
            end
        end
    end

    if total_atoms > 0
        com ./= total_atoms

        # Subtract center of mass from all motif atoms
        for l in 1:L
            for a in 1:n_atoms
                if motif_mask[a, l]
                    x_motif[:, a, l] .-= com
                end
            end
        end
    end
end
