# Triangular multiplicative updates for pair representation
# Port of openfold/model/triangular_multiplicative_update.py
# Implements Algorithms 11 (outgoing) and 12 (incoming) from AlphaFold

using Flux

"""
    TriangleMultiplication(pair_dim::Int, c_hidden::Int, mode::Symbol)

Triangular multiplicative update for pair representations.
`mode` is `:outgoing` (Algorithm 11) or `:incoming` (Algorithm 12).

# Forward
Given pair representation z [D, L, L, B] and pair_mask [L, L, B]:
1. z_normed = layer_norm_in(z)
2. a = linear_a_p(z_normed) .* sigmoid.(linear_a_g(z_normed)) .* mask
3. b = linear_b_p(z_normed) .* sigmoid.(linear_b_g(z_normed)) .* mask
4. Outgoing: x[i,j] = sum_k a[i,k] * b[j,k]  (contract over k)
   Incoming: x[k,j] = sum_i a[i,k] * b[i,j]  (contract over i)
5. x = layer_norm_out(x)
6. x = linear_z(x) .* sigmoid.(linear_g(z_normed))
"""
struct TriangleMultiplication
    layer_norm_in::PyTorchLayerNorm
    linear_a_p::Dense
    linear_a_g::Dense
    linear_b_p::Dense
    linear_b_g::Dense
    linear_g::Dense
    linear_z::Dense
    layer_norm_out::PyTorchLayerNorm
    mode::Symbol  # :outgoing or :incoming
end

Flux.@layer TriangleMultiplication

function TriangleMultiplication(pair_dim::Int; c_hidden::Int=0, mode::Symbol=:outgoing)
    if c_hidden <= 0
        c_hidden = min(pair_dim, 196)
    end
    @assert mode in (:outgoing, :incoming) "mode must be :outgoing or :incoming"

    layer_norm_in = PyTorchLayerNorm(pair_dim)
    linear_a_p = Dense(pair_dim => c_hidden)
    linear_a_g = Dense(pair_dim => c_hidden)
    linear_b_p = Dense(pair_dim => c_hidden)
    linear_b_g = Dense(pair_dim => c_hidden)
    linear_g = Dense(pair_dim => pair_dim)
    linear_z = Dense(c_hidden => pair_dim)
    layer_norm_out = PyTorchLayerNorm(c_hidden)

    return TriangleMultiplication(
        layer_norm_in, linear_a_p, linear_a_g, linear_b_p, linear_b_g,
        linear_g, linear_z, layer_norm_out, mode
    )
end

function (m::TriangleMultiplication)(z, pair_mask)
    # z: [D, L, L, B]
    # pair_mask: [L, L, B]
    D, L, _, B = size(z)
    mask_exp = reshape(pair_mask, 1, L, L, B)  # [1, L, L, B]

    # Layer norm input
    z_normed = m.layer_norm_in(z)

    # Gated projections
    # Reshape for Dense: [D, L*L*B] → project → reshape back to [c_hidden, L, L, B]
    z_flat = reshape(z_normed, D, L * L * B)

    a = reshape(m.linear_a_p(z_flat), :, L, L, B) .* NNlib.sigmoid.(reshape(m.linear_a_g(z_flat), :, L, L, B))
    a = a .* mask_exp  # [c_hidden, L, L, B]

    b = reshape(m.linear_b_p(z_flat), :, L, L, B) .* NNlib.sigmoid.(reshape(m.linear_b_g(z_flat), :, L, L, B))
    b = b .* mask_exp  # [c_hidden, L, L, B]

    c_hidden = size(a, 1)

    # Triangle contraction using batched_mul
    # NNlib.batched_mul: [m, k, batch] × [k, n, batch] → [m, n, batch]
    # We put (c_hidden, B) into the batch dimension to keep channels independent.
    if m.mode == :outgoing
        # x[c,i,j,b] = sum_k a[c,i,k,b] * b[c,j,k,b]
        # a: [c_h, L_i, L_k, B] → [L_i, L_k, c_h, B] → [L_i, L_k, c_h*B]
        # b: [c_h, L_j, L_k, B] → [L_k, L_j, c_h, B] → [L_k, L_j, c_h*B]
        a_perm = permutedims(a, (2, 3, 1, 4))  # [L_i, L_k, c_h, B]
        b_perm = permutedims(b, (3, 2, 1, 4))  # [L_k, L_j, c_h, B]
        a_mat = reshape(a_perm, L, L, c_hidden * B)
        b_mat = reshape(b_perm, L, L, c_hidden * B)
        x_mat = NNlib.batched_mul(a_mat, b_mat)  # [L_i, L_j, c_h*B]
        x = reshape(x_mat, L, L, c_hidden, B)
        x = permutedims(x, (3, 1, 2, 4))  # [c_h, L_i, L_j, B]
    else  # :incoming
        # x[c,k,j,b] = sum_i a[c,i,k,b] * b[c,i,j,b]
        # a: [c_h, L_i, L_k, B] → [L_k, L_i, c_h, B] → [L_k, L_i, c_h*B]
        # b: [c_h, L_i, L_j, B] → [L_i, L_j, c_h, B] → [L_i, L_j, c_h*B]
        a_perm = permutedims(a, (3, 2, 1, 4))  # [L_k, L_i, c_h, B]
        b_perm = permutedims(b, (2, 3, 1, 4))  # [L_i, L_j, c_h, B]
        a_mat = reshape(a_perm, L, L, c_hidden * B)
        b_mat = reshape(b_perm, L, L, c_hidden * B)
        x_mat = NNlib.batched_mul(a_mat, b_mat)  # [L_k, L_j, c_h*B]
        x = reshape(x_mat, L, L, c_hidden, B)
        x = permutedims(x, (3, 1, 2, 4))  # [c_h, L_k, L_j, B]
    end

    # Layer norm on contracted output
    x = m.layer_norm_out(x)

    # Output projection with gate
    x_flat = reshape(x, c_hidden, L * L * B)
    x = reshape(m.linear_z(x_flat), :, L, L, B)  # [pair_dim, L, L, B]

    # Gate from original (normed) input
    g = NNlib.sigmoid.(reshape(m.linear_g(z_flat), :, L, L, B))  # [pair_dim, L, L, B]
    x = x .* g

    return x
end

"""
    PairTransition(pair_dim::Int; n::Int=2)

Simple feedforward transition for pair representation.
Implements Algorithm 15 from AlphaFold.

Forward: z → LayerNorm → Dense(expand) → ReLU → Dense(contract) → mask
"""
struct PairTransition
    layer_norm::PyTorchLayerNorm
    linear_1::Dense
    linear_2::Dense
end

Flux.@layer PairTransition

function PairTransition(pair_dim::Int; n::Int=2)
    c_hidden = n * pair_dim
    layer_norm = PyTorchLayerNorm(pair_dim)
    linear_1 = Dense(pair_dim => c_hidden)
    linear_2 = Dense(c_hidden => pair_dim)
    return PairTransition(layer_norm, linear_1, linear_2)
end

function (m::PairTransition)(z, pair_mask)
    # z: [D, L, L, B]
    # pair_mask: [L, L, B]
    D, L, _, B = size(z)
    mask_exp = reshape(pair_mask, 1, L, L, B)

    z = m.layer_norm(z)

    # Flatten, project, reshape
    z_flat = reshape(z, D, L * L * B)
    h = NNlib.relu.(m.linear_1(z_flat))  # [c_hidden, L*L*B]
    out = m.linear_2(h)                   # [D, L*L*B]
    out = reshape(out, D, L, L, B)

    return out .* mask_exp
end
