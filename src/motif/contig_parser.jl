# Contig string parser for motif scaffolding
# Port of generate_motif_indices from proteinfoundation/utils/motif_utils.py

"""
    ContigSegment

A segment in a contig string: either a scaffold region or a motif region.
"""
abstract type ContigSegment end

"""
    ScaffoldSegment(min_len::Int, max_len::Int)

Variable-length scaffold region. If min_len == max_len, it's a fixed-length scaffold.
"""
struct ScaffoldSegment <: ContigSegment
    min_len::Int
    max_len::Int
end

"""
    MotifSegment(chain::Char, start_res::Int, end_res::Int)

Fixed motif region from a specific chain and residue range in the source PDB.
"""
struct MotifSegment <: ContigSegment
    chain::Char
    start_res::Int
    end_res::Int
end

motif_length(m::MotifSegment) = m.end_res - m.start_res + 1

"""
    parse_contig(contig_string::String) -> Vector{ContigSegment}

Parse a contig string into a vector of segments.

# Format
- Scaffold: `"10-40"` (variable length 10 to 40) or `"15"` (fixed length 15)
- Motif: `"A163-181"` (chain A, residues 163-181)
- Separator: `/`

# Examples
```julia
parse_contig("10-40/A163-181/10-40")
# → [ScaffoldSegment(10,40), MotifSegment('A',163,181), ScaffoldSegment(10,40)]

parse_contig("15/B1-20/10-25/B30-50/15")
# → [ScaffoldSegment(15,15), MotifSegment('B',1,20), ScaffoldSegment(10,25),
#    MotifSegment('B',30,50), ScaffoldSegment(15,15)]
```
"""
function parse_contig(contig_string::String)
    segments = ContigSegment[]
    parts = split(strip(contig_string), '/')

    for part in parts
        part = strip(String(part))
        isempty(part) && continue

        if isletter(part[1])
            # Motif segment: e.g. "A163-181"
            chain = part[1]
            range_str = part[2:end]
            if contains(range_str, '-')
                start_str, end_str = split(range_str, '-')
                push!(segments, MotifSegment(chain, parse(Int, start_str), parse(Int, end_str)))
            else
                # Single residue
                res = parse(Int, range_str)
                push!(segments, MotifSegment(chain, res, res))
            end
        else
            # Scaffold segment: e.g. "10-40" or "15"
            if contains(part, '-')
                min_str, max_str = split(part, '-')
                push!(segments, ScaffoldSegment(parse(Int, min_str), parse(Int, max_str)))
            else
                len = parse(Int, part)
                push!(segments, ScaffoldSegment(len, len))
            end
        end
    end

    return segments
end

"""
    generate_scaffold_lengths(segments::Vector{ContigSegment};
                               min_length::Int=50, max_length::Int=512,
                               n_samples::Int=1) -> Vector{Vector{Int}}

Sample valid scaffold lengths that satisfy total length constraints.

Returns a vector of n_samples, each being a vector of actual lengths for each segment
(motif segments keep their fixed length, scaffold segments get sampled).
"""
function generate_scaffold_lengths(segments::Vector{<:ContigSegment};
                                    min_length::Int=50, max_length::Int=512,
                                    n_samples::Int=1)
    # Compute fixed motif length and scaffold bounds
    motif_total = sum(s isa MotifSegment ? motif_length(s) : 0 for s in segments)
    scaffold_min = sum(s isa ScaffoldSegment ? s.min_len : 0 for s in segments)
    scaffold_max = sum(s isa ScaffoldSegment ? s.max_len : 0 for s in segments)

    # Effective bounds for total length
    total_min = max(min_length, motif_total + scaffold_min)
    total_max = min(max_length, motif_total + scaffold_max)

    if total_min > total_max
        error("Cannot satisfy length constraints: min=$total_min > max=$total_max " *
              "(motif=$motif_total, scaffold_min=$scaffold_min, scaffold_max=$scaffold_max)")
    end

    results = Vector{Vector{Int}}()

    for _ in 1:n_samples
        # Sample total length
        total_length = rand(total_min:total_max)
        scaffold_budget = total_length - motif_total

        # Distribute scaffold budget across scaffold segments
        lengths = Int[]
        scaffold_indices = Int[]
        for (i, seg) in enumerate(segments)
            if seg isa MotifSegment
                push!(lengths, motif_length(seg))
            else
                push!(lengths, seg.min_len)
                push!(scaffold_indices, i)
            end
        end

        # Distribute remaining budget
        remaining = scaffold_budget - sum(lengths[i] for i in scaffold_indices; init=0)
        for idx in scaffold_indices
            seg = segments[idx]
            extra = min(remaining, seg.max_len - seg.min_len)
            if extra > 0
                add = rand(0:extra)
                lengths[idx] += add
                remaining -= add
            end
        end

        # Distribute any remaining budget evenly
        while remaining > 0 && !isempty(scaffold_indices)
            for idx in scaffold_indices
                seg = segments[idx]
                if lengths[idx] < seg.max_len && remaining > 0
                    lengths[idx] += 1
                    remaining -= 1
                end
            end
        end

        push!(results, lengths)
    end

    return results
end

"""
    compute_motif_indices(segments::Vector{<:ContigSegment},
                           lengths::Vector{Int}) -> Vector{Int}

Given segments and their sampled lengths, compute the 1-indexed positions
where motif residues are placed in the full-length protein.
"""
function compute_motif_indices(segments::Vector{<:ContigSegment},
                                lengths::Vector{Int})
    indices = Int[]
    pos = 0
    for (i, seg) in enumerate(segments)
        if seg isa MotifSegment
            for j in 1:lengths[i]
                push!(indices, pos + j)
            end
        end
        pos += lengths[i]
    end
    return indices
end
