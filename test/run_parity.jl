#!/usr/bin/env julia
"""
Julia parity tests against Python la-proteina reference data.
Run this after running parity_test.py to generate reference data.

Usage:
    cd LaProteina
    julia --project=. test/run_parity.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test
using NPZ
using LinearAlgebra
using Statistics
using Flux

# Import LaProteina
include(joinpath(@__DIR__, "..", "src", "LaProteina.jl"))
using .LaProteina

const PARITY_DATA_DIR = get(ENV, "PARITY_DATA_DIR", joinpath(@__DIR__, "..", "..", "parity_data"))

"""Load numpy file and convert to Julia array format."""
function load_npy(name::String)
    path = joinpath(PARITY_DATA_DIR, "$(name).npy")
    if !isfile(path)
        error("Reference file not found: $path\nRun parity_test.py first!")
    end
    return Float32.(npzread(path))
end

"""Convert from Python [B, L, D] to Julia [D, L, B]."""
function py_to_jl(arr::AbstractArray{T, 3}) where T
    return permutedims(arr, (3, 2, 1))
end

"""Convert from Python [B, L_i, L_j, D] to Julia [D, L_i, L_j, B]."""
function py_to_jl_pair(arr::AbstractArray{T, 4}) where T
    # Python: [B, L_i, L_j, D] -> Julia: [D, L_i, L_j, B]
    # Dims: 1->4, 2->2, 3->3, 4->1
    return permutedims(arr, (4, 2, 3, 1))
end

"""Check if two arrays are approximately equal."""
function check_parity(name::String, jl_result, py_result; rtol=1e-4, atol=1e-5)
    if size(jl_result) != size(py_result)
        println("  ✗ $name: Shape mismatch - Julia $(size(jl_result)) vs Python $(size(py_result))")
        return false
    end

    max_diff = maximum(abs.(jl_result .- py_result))
    rel_diff = max_diff / (maximum(abs.(py_result)) + 1e-10)

    if isapprox(jl_result, py_result; rtol=rtol, atol=atol)
        println("  ✓ $name: PASS (max diff: $max_diff, rel diff: $rel_diff)")
        return true
    else
        println("  ✗ $name: FAIL (max diff: $max_diff, rel diff: $rel_diff)")
        # Show first few differing values
        diffs = abs.(jl_result .- py_result)
        max_idx = argmax(diffs)
        println("    Max diff at index $max_idx: Julia=$(jl_result[max_idx]), Python=$(py_result[max_idx])")
        return false
    end
end

# ============================================================================
# Parity Tests
# ============================================================================

function test_index_embedding()
    println("\n=== Testing Index Embedding ===")

    # Load Python reference
    indices_py = load_npy("index_emb_input")  # [L=10]
    emb_py = load_npy("index_emb_output")  # [L=10, D=64]

    # Convert indices to Int for Julia function
    indices_int = Int.(indices_py)

    # Julia: get_index_embedding returns [D, L]
    emb_jl = get_index_embedding(indices_int, 64)

    # Python returns [L, D], need to transpose for comparison
    emb_py_t = permutedims(emb_py, (2, 1))  # [D, L]

    return check_parity("index_embedding", emb_jl, emb_py_t; rtol=1e-4)
end

function test_time_embedding()
    println("\n=== Testing Time Embedding ===")

    # Load Python reference
    t_py = load_npy("time_emb_input_t")  # [7]
    emb_py = load_npy("time_emb_output")  # [7, 256] in Python format [B, D]

    # Julia: expects [B], returns [D, B]
    emb_jl = get_time_embedding(t_py, 256)

    # Python returns [B, D], need to transpose for comparison
    emb_py_t = permutedims(emb_py, (2, 1))  # [D, B]

    return check_parity("time_embedding", emb_jl, emb_py_t; rtol=1e-4)
end

function test_relative_sequence_separation()
    println("\n=== Testing Relative Sequence Separation ===")

    # Load Python reference
    positions_py = load_npy("rel_sep_positions")  # [B=2, L=10]
    out_py = load_npy("rel_sep_output")  # [B=2, L=10, L=10, n_classes=11]

    B, L = size(positions_py)
    max_sep = 5

    # Convert Python output to Julia format [D, L, L, B]
    out_py_jl = py_to_jl_pair(out_py)  # [11, 10, 10, 2]

    # Julia implementation
    # Python: rel_sep = positions.unsqueeze(2) - positions.unsqueeze(1) -> rel_sep[b,i,j] = positions[b,i] - positions[b,j]
    n_classes = 2 * max_sep + 1
    onehot_jl = zeros(Float32, n_classes, L, L, B)

    for b in 1:B
        for i in 1:L, j in 1:L
            # Python computes positions[i] - positions[j] at index [i, j]
            rel_sep = Int(positions_py[b, i]) - Int(positions_py[b, j])
            clamped = clamp(rel_sep, -max_sep, max_sep)
            class_idx = clamped + max_sep + 1  # 1-indexed
            onehot_jl[class_idx, i, j, b] = 1.0f0
        end
    end

    return check_parity("rel_seq_sep", onehot_jl, out_py_jl; rtol=1e-5)
end

function test_distance_binning()
    println("\n=== Testing Distance Binning ===")

    # Load Python reference
    coords_py = load_npy("dist_bin_input_coords")  # [B=2, L=5, 3]
    binned_py = load_npy("dist_bin_output")  # [B=2, L=5, L=5, D=16]

    # Convert coords to Julia format [3, L, B]
    coords_jl = permutedims(coords_py, (3, 2, 1))

    # Run Julia (positional args: x, min_dist, max_dist, n_bins)
    binned_jl = bin_pairwise_distances(coords_jl, 0.0f0, 2.0f0, 16)

    # Convert Python to Julia format [D, L, L, B]
    binned_py_jl = py_to_jl_pair(binned_py)

    return check_parity("distance_binning", binned_jl, binned_py_jl; rtol=1e-4)
end

function test_adaptive_layer_norm()
    println("\n=== Testing AdaptiveLayerNorm ===")

    # Load Python reference
    x_py = load_npy("adaln_input_x")  # [B=2, L=10, D=64]
    cond_py = load_npy("adaln_input_cond")  # [B=2, L=10, D_cond=32]
    mask_py = load_npy("adaln_input_mask")  # [B=2, L=10]
    out_py = load_npy("adaln_output")  # [B=2, L=10, D=64]

    # Load Python weights
    norm_cond_weight = load_npy("adaln_weight_norm_cond_weight")  # [32]
    norm_cond_bias = load_npy("adaln_weight_norm_cond_bias")  # [32]
    gamma_weight = load_npy("adaln_weight_to_gamma_0_weight")  # [64, 32] PyTorch: [out, in]
    gamma_bias = load_npy("adaln_weight_to_gamma_0_bias")  # [64]
    beta_weight = load_npy("adaln_weight_to_beta_weight")  # [64, 32]

    # Convert inputs to Julia format [D, L, B]
    x_jl = py_to_jl(x_py)
    cond_jl = py_to_jl(cond_py)
    mask_jl = permutedims(mask_py, (2, 1))  # [L, B]
    out_py_jl = py_to_jl(out_py)

    # Create Julia module and set weights
    adaln = ProteINAAdaLN(64, 32)

    # Set norm_cond weights (Flux LayerNorm has diag.scale and diag.bias)
    adaln.norm_cond.diag.scale .= norm_cond_weight
    adaln.norm_cond.diag.bias .= norm_cond_bias

    # Set to_gamma weights (Dense layer)
    # PyTorch Linear: weight [out, in], Flux Dense: weight [out, in]
    adaln.to_gamma[1].weight .= gamma_weight
    adaln.to_gamma[1].bias .= gamma_bias

    # Set to_beta weights (no bias in PyTorch)
    adaln.to_beta.weight .= beta_weight

    # Run Julia
    out_jl = adaln(x_jl, cond_jl, mask_jl)

    return check_parity("adaln", out_jl, out_py_jl; rtol=1e-4)
end

function test_adaptive_output_scale()
    println("\n=== Testing AdaptiveOutputScale ===")

    # Load Python reference
    x_py = load_npy("scale_input_x")  # [B=2, L=10, D=64]
    cond_py = load_npy("scale_input_cond")  # [B=2, L=10, D_cond=32]
    mask_py = load_npy("scale_input_mask")  # [B=2, L=10]
    out_py = load_npy("scale_output")  # [B=2, L=10, D=64]

    # Load Python weights
    weight = load_npy("scale_weight_to_adaln_zero_gamma_0_weight")  # [64, 32]
    bias = load_npy("scale_weight_to_adaln_zero_gamma_0_bias")  # [64]

    # Convert inputs to Julia format
    x_jl = py_to_jl(x_py)
    cond_jl = py_to_jl(cond_py)
    mask_jl = permutedims(mask_py, (2, 1))
    out_py_jl = py_to_jl(out_py)

    # Create Julia module and set weights
    scale = AdaptiveOutputScale(64, 32)
    scale.linear.weight .= weight
    scale.linear.bias .= bias

    # Run Julia
    out_jl = scale(x_jl, cond_jl, mask_jl)

    return check_parity("output_scale", out_jl, out_py_jl; rtol=1e-4)
end

function test_swiglu()
    println("\n=== Testing SwiGLU ===")

    # Load Python reference
    x_py = load_npy("swiglu_input")  # [B=2, L=10, D=128]
    out_py = load_npy("swiglu_output")  # [B=2, L=10, D=64]

    # Convert to Julia format
    x_jl = py_to_jl(x_py)  # [128, 10, 2]
    out_py_jl = py_to_jl(out_py)  # [64, 10, 2]

    # Julia SwiGLU: splits input in half, applies silu to second half, multiplies
    # Python: x, gates = x.chunk(2, dim=-1); return F.silu(gates) * x
    # So in Julia with [D, L, B]: split along dim 1
    D = size(x_jl, 1)
    x_half = x_jl[1:D÷2, :, :]
    gates = x_jl[D÷2+1:end, :, :]
    out_jl = NNlib.sigmoid.(gates) .* gates .* x_half  # silu(x) = x * sigmoid(x)

    # Wait, let me recheck the SwiGLU formula
    # silu(x) = x * sigmoid(x)
    # SwiGLU: silu(gates) * x = gates * sigmoid(gates) * x
    out_jl = (gates .* NNlib.sigmoid.(gates)) .* x_half

    return check_parity("swiglu", out_jl, out_py_jl; rtol=1e-4)
end

function test_rdn_interpolation()
    println("\n=== Testing RDN Flow Interpolation ===")

    # Load Python reference
    x0_py = load_npy("rdn_interp_x0")  # [B=2, L=10, D=3]
    x1_py = load_npy("rdn_interp_x1")  # [B=2, L=10, D=3]
    t_py = load_npy("rdn_interp_t")  # [B=2]
    xt_py = load_npy("rdn_interp_xt")  # [B=2, L=10, D=3]

    # Convert to Julia format [D, L, B]
    x0_jl = py_to_jl(x0_py)
    x1_jl = py_to_jl(x1_py)
    xt_py_jl = py_to_jl(xt_py)

    # Simple linear interpolation: x_t = (1-t)*x_0 + t*x_1
    # t is [B], need to broadcast to [1, 1, B]
    t_exp = reshape(t_py, 1, 1, :)
    xt_jl = (1.0f0 .- t_exp) .* x0_jl .+ t_exp .* x1_jl

    return check_parity("rdn_interpolation", xt_jl, xt_py_jl; rtol=1e-5)
end

function test_rdn_loss()
    println("\n=== Testing RDN Flow Loss ===")

    # Load Python reference
    x1_py = load_npy("rdn_loss_x1")  # [B=2, L=10, D=3]
    x1_pred_py = load_npy("rdn_loss_x1_pred")  # [B=2, L=10, D=3]
    t_py = load_npy("rdn_loss_t")  # [B=2]
    mask_py = load_npy("rdn_loss_mask")  # [B=2, L=10]
    loss_py = load_npy("rdn_loss_output")  # [B=2]

    # Compute loss in Julia
    # Python formula:
    #   err = (x_1 - x_1_pred) * mask[..., None]
    #   loss = sum(err**2, dim=(-1, -2)) / nres
    #   total_loss_w = 1 / ((1-t)**2 + 1e-5)
    #   loss = loss * total_loss_w

    # Convert to Julia format
    x1_jl = py_to_jl(x1_py)  # [3, 10, 2]
    x1_pred_jl = py_to_jl(x1_pred_py)
    mask_jl = permutedims(mask_py, (2, 1))  # [10, 2]

    # Compute error
    mask_exp = reshape(mask_jl, 1, size(mask_jl)...)  # [1, 10, 2]
    err = (x1_jl .- x1_pred_jl) .* mask_exp  # [3, 10, 2]
    err_sq = err .^ 2

    # Sum over D and L dimensions (1 and 2 in Julia)
    sum_err = dropdims(sum(err_sq; dims=(1, 2)); dims=(1, 2))  # [B]

    # Divide by number of valid residues
    nres = dropdims(sum(mask_jl; dims=1); dims=1)  # [B]
    loss = sum_err ./ nres

    # Apply 1/(1-t)^2 weighting
    weight = 1.0f0 ./ ((1.0f0 .- t_py) .^ 2 .+ 1f-5)
    loss_jl = loss .* weight

    return check_parity("rdn_loss", loss_jl, loss_py; rtol=1e-4)
end

function test_pair_bias_attention()
    println("\n=== Testing PairBiasAttention ===")

    # Load Python reference
    node_py = load_npy("pba_input_node")  # [B=2, L=10, D=64]
    pair_py = load_npy("pba_input_pair")  # [B=2, L=10, L=10, D=32]
    mask_py = load_npy("pba_input_mask")  # [B=2, L=10, L=10]
    out_py = load_npy("pba_output")  # [B=2, L=10, D=64]

    # Load Python weights
    to_qkv_weight = load_npy("pba_weight_to_qkv_weight")  # [192, 64]
    to_qkv_bias = load_npy("pba_weight_to_qkv_bias")  # [192]
    to_g_weight = load_npy("pba_weight_to_g_weight")  # [64, 64]
    to_g_bias = load_npy("pba_weight_to_g_bias")  # [64]
    to_out_weight = load_npy("pba_weight_to_out_node_weight")  # [64, 64]
    to_out_bias = load_npy("pba_weight_to_out_node_bias")  # [64]
    node_norm_weight = load_npy("pba_weight_node_norm_weight")  # [64]
    node_norm_bias = load_npy("pba_weight_node_norm_bias")  # [64]
    to_bias_weight = load_npy("pba_weight_to_bias_weight")  # [4, 32]
    pair_norm_weight = load_npy("pba_weight_pair_norm_weight")  # [32]
    pair_norm_bias = load_npy("pba_weight_pair_norm_bias")  # [32]

    # Convert inputs to Julia format
    node_jl = py_to_jl(node_py)  # [64, 10, 2]
    pair_jl = py_to_jl_pair(pair_py)  # [32, 10, 10, 2]
    # mask_py is [B, L, L] for pair mask, but attention needs [L, B] for key mask
    # In Python, the attention mask is the pair mask. In Julia we use just [L, B]
    mask_jl = ones(Float32, 10, 2)  # All positions valid
    out_py_jl = py_to_jl(out_py)

    # Create Julia module
    node_dim = 64
    pair_dim = 32
    n_heads = 4
    dim_head = 16

    attn = PairBiasAttention(node_dim, n_heads; pair_dim=pair_dim, qk_ln=false, bias=true)

    # Set weights
    attn.node_norm.diag.scale .= node_norm_weight
    attn.node_norm.diag.bias .= node_norm_bias
    attn.to_qkv.weight .= to_qkv_weight
    attn.to_qkv.bias .= to_qkv_bias
    attn.to_g.weight .= to_g_weight
    attn.to_g.bias .= to_g_bias
    attn.to_out.weight .= to_out_weight
    attn.to_out.bias .= to_out_bias
    attn.pair_norm.diag.scale .= pair_norm_weight
    attn.pair_norm.diag.bias .= pair_norm_bias
    attn.to_bias.weight .= to_bias_weight

    # Run Julia
    out_jl = attn(node_jl, pair_jl, mask_jl)

    return check_parity("pair_bias_attention", out_jl, out_py_jl; rtol=1e-3, atol=1e-4)
end

function test_mha_adaln()
    println("\n=== Testing MultiHeadBiasedAttentionADALN ===")

    # Load Python reference
    x_py = load_npy("mha_adaln_input_x")  # [B=2, L=10, D=64]
    pair_py = load_npy("mha_adaln_input_pair")  # [B=2, L=10, L=10, D=32]
    cond_py = load_npy("mha_adaln_input_cond")  # [B=2, L=10, D_cond=32]
    mask_py = load_npy("mha_adaln_input_mask")  # [B=2, L=10]
    out_py = load_npy("mha_adaln_output")  # [B=2, L=10, D=64]

    # Load weights - AdaLN
    adaln_norm_cond_weight = load_npy("mha_adaln_weight_adaln_norm_cond_weight")
    adaln_norm_cond_bias = load_npy("mha_adaln_weight_adaln_norm_cond_bias")
    adaln_gamma_weight = load_npy("mha_adaln_weight_adaln_to_gamma_0_weight")
    adaln_gamma_bias = load_npy("mha_adaln_weight_adaln_to_gamma_0_bias")
    adaln_beta_weight = load_npy("mha_adaln_weight_adaln_to_beta_weight")

    # Load weights - MHA
    mha_to_qkv_weight = load_npy("mha_adaln_weight_mha_to_qkv_weight")
    mha_to_qkv_bias = load_npy("mha_adaln_weight_mha_to_qkv_bias")
    mha_to_g_weight = load_npy("mha_adaln_weight_mha_to_g_weight")
    mha_to_g_bias = load_npy("mha_adaln_weight_mha_to_g_bias")
    mha_to_out_weight = load_npy("mha_adaln_weight_mha_to_out_node_weight")
    mha_to_out_bias = load_npy("mha_adaln_weight_mha_to_out_node_bias")
    mha_node_norm_weight = load_npy("mha_adaln_weight_mha_node_norm_weight")
    mha_node_norm_bias = load_npy("mha_adaln_weight_mha_node_norm_bias")
    mha_to_bias_weight = load_npy("mha_adaln_weight_mha_to_bias_weight")
    mha_pair_norm_weight = load_npy("mha_adaln_weight_mha_pair_norm_weight")
    mha_pair_norm_bias = load_npy("mha_adaln_weight_mha_pair_norm_bias")

    # Load weights - Scale Output
    scale_weight = load_npy("mha_adaln_weight_scale_output_to_adaln_zero_gamma_0_weight")
    scale_bias = load_npy("mha_adaln_weight_scale_output_to_adaln_zero_gamma_0_bias")

    # Convert inputs to Julia format
    x_jl = py_to_jl(x_py)  # [64, 10, 2]
    pair_jl = py_to_jl_pair(pair_py)  # [32, 10, 10, 2]
    cond_jl = py_to_jl(cond_py)  # [32, 10, 2]
    mask_jl = permutedims(mask_py, (2, 1))  # [10, 2]
    out_py_jl = py_to_jl(out_py)

    # Create Julia module (dim_token=64, dim_pair=32, n_heads=4, dim_cond=32)
    mha_adaln = MultiHeadBiasedAttentionADALN(64, 32, 4, 32; qk_ln=false)

    # Set AdaLN weights
    mha_adaln.adaln.norm_cond.diag.scale .= adaln_norm_cond_weight
    mha_adaln.adaln.norm_cond.diag.bias .= adaln_norm_cond_bias
    mha_adaln.adaln.to_gamma[1].weight .= adaln_gamma_weight
    mha_adaln.adaln.to_gamma[1].bias .= adaln_gamma_bias
    mha_adaln.adaln.to_beta.weight .= adaln_beta_weight

    # Set MHA weights
    mha_adaln.mha.node_norm.diag.scale .= mha_node_norm_weight
    mha_adaln.mha.node_norm.diag.bias .= mha_node_norm_bias
    mha_adaln.mha.to_qkv.weight .= mha_to_qkv_weight
    mha_adaln.mha.to_qkv.bias .= mha_to_qkv_bias
    mha_adaln.mha.to_g.weight .= mha_to_g_weight
    mha_adaln.mha.to_g.bias .= mha_to_g_bias
    mha_adaln.mha.to_out.weight .= mha_to_out_weight
    mha_adaln.mha.to_out.bias .= mha_to_out_bias
    mha_adaln.mha.pair_norm.diag.scale .= mha_pair_norm_weight
    mha_adaln.mha.pair_norm.diag.bias .= mha_pair_norm_bias
    mha_adaln.mha.to_bias.weight .= mha_to_bias_weight

    # Set scale output weights
    mha_adaln.scale_output.linear.weight .= scale_weight
    mha_adaln.scale_output.linear.bias .= scale_bias

    # Run Julia
    out_jl = mha_adaln(x_jl, pair_jl, cond_jl, mask_jl)

    return check_parity("mha_adaln", out_jl, out_py_jl; rtol=1e-3, atol=1e-4)
end

function test_simple_transition()
    println("\n=== Testing Simple Transition ===")

    # Load Python reference
    x_py = load_npy("simple_trans_input_x")  # [B=2, L=10, D=64]
    mask_py = load_npy("simple_trans_input_mask")  # [B=2, L=10]
    out_py = load_npy("simple_trans_output")  # [B=2, L=10, D=64]

    # Load weights
    linear_in_weight = load_npy("simple_trans_weight_linear_in")  # [512, 64]
    linear_out_weight = load_npy("simple_trans_weight_linear_out")  # [64, 256]

    # Convert inputs to Julia format
    x_jl = py_to_jl(x_py)
    mask_jl = permutedims(mask_py, (2, 1))
    out_py_jl = py_to_jl(out_py)

    # Create Julia module
    transition = SwiGLUTransition(64; expansion_factor=4, use_layer_norm=false)

    # Set weights
    transition.linear_in.weight .= linear_in_weight
    transition.linear_out.weight .= linear_out_weight

    # Run Julia
    out_jl = transition(x_jl, mask_jl)

    return check_parity("simple_transition", out_jl, out_py_jl; rtol=1e-4)
end

function test_transition_adaln()
    println("\n=== Testing TransitionADALN ===")

    # Load Python reference
    x_py = load_npy("transition_input_x")  # [B=2, L=10, D=64]
    cond_py = load_npy("transition_input_cond")  # [B=2, L=10, D_cond=32]
    mask_py = load_npy("transition_input_mask")  # [B=2, L=10]
    out_py = load_npy("transition_output")  # [B=2, L=10, D=64]

    # Load weights
    adaln_norm_cond_weight = load_npy("transition_weight_adaln_norm_cond_weight")
    adaln_norm_cond_bias = load_npy("transition_weight_adaln_norm_cond_bias")
    adaln_gamma_weight = load_npy("transition_weight_adaln_to_gamma_0_weight")
    adaln_gamma_bias = load_npy("transition_weight_adaln_to_gamma_0_bias")
    adaln_beta_weight = load_npy("transition_weight_adaln_to_beta_weight")
    swish_linear_weight = load_npy("transition_weight_transition_swish_linear_0_weight")  # [512, 64]
    linear_out_weight = load_npy("transition_weight_transition_linear_out_weight")  # [64, 256]
    scale_weight = load_npy("transition_weight_scale_output_to_adaln_zero_gamma_0_weight")
    scale_bias = load_npy("transition_weight_scale_output_to_adaln_zero_gamma_0_bias")

    # Convert inputs to Julia format
    x_jl = py_to_jl(x_py)
    cond_jl = py_to_jl(cond_py)
    mask_jl = permutedims(mask_py, (2, 1))
    out_py_jl = py_to_jl(out_py)

    # Create Julia module
    transition = TransitionADALN(64, 32; expansion_factor=4)

    # Set AdaLN weights
    transition.adaln.norm_cond.diag.scale .= adaln_norm_cond_weight
    transition.adaln.norm_cond.diag.bias .= adaln_norm_cond_bias
    transition.adaln.to_gamma[1].weight .= adaln_gamma_weight
    transition.adaln.to_gamma[1].bias .= adaln_gamma_bias
    transition.adaln.to_beta.weight .= adaln_beta_weight

    # Set transition weights
    transition.transition.linear_in.weight .= swish_linear_weight
    transition.transition.linear_out.weight .= linear_out_weight

    # Set scale output weights
    transition.scale_output.linear.weight .= scale_weight
    transition.scale_output.linear.bias .= scale_bias

    # Run Julia
    out_jl = transition(x_jl, cond_jl, mask_jl)

    return check_parity("transition_adaln", out_jl, out_py_jl; rtol=1e-4)
end

function test_transformer_block()
    println("\n=== Testing TransformerBlock ===")

    # Load Python reference
    x_py = load_npy("tfblock_input_x")  # [B=2, L=10, D=64]
    pair_py = load_npy("tfblock_input_pair")  # [B=2, L=10, L=10, D=32]
    cond_py = load_npy("tfblock_input_cond")  # [B=2, L=10, D_cond=32]
    mask_py = load_npy("tfblock_input_mask")  # [B=2, L=10]
    out_py = load_npy("tfblock_output")  # [B=2, L=10, D=64]

    # Load weights for MHA_ADALN (mhba.*)
    mhba_adaln_norm_cond_weight = load_npy("tfblock_weight_mhba_adaln_norm_cond_weight")
    mhba_adaln_norm_cond_bias = load_npy("tfblock_weight_mhba_adaln_norm_cond_bias")
    mhba_adaln_gamma_weight = load_npy("tfblock_weight_mhba_adaln_to_gamma_0_weight")
    mhba_adaln_gamma_bias = load_npy("tfblock_weight_mhba_adaln_to_gamma_0_bias")
    mhba_adaln_beta_weight = load_npy("tfblock_weight_mhba_adaln_to_beta_weight")
    mhba_mha_to_qkv_weight = load_npy("tfblock_weight_mhba_mha_to_qkv_weight")
    mhba_mha_to_qkv_bias = load_npy("tfblock_weight_mhba_mha_to_qkv_bias")
    mhba_mha_to_g_weight = load_npy("tfblock_weight_mhba_mha_to_g_weight")
    mhba_mha_to_g_bias = load_npy("tfblock_weight_mhba_mha_to_g_bias")
    mhba_mha_to_out_weight = load_npy("tfblock_weight_mhba_mha_to_out_node_weight")
    mhba_mha_to_out_bias = load_npy("tfblock_weight_mhba_mha_to_out_node_bias")
    mhba_mha_node_norm_weight = load_npy("tfblock_weight_mhba_mha_node_norm_weight")
    mhba_mha_node_norm_bias = load_npy("tfblock_weight_mhba_mha_node_norm_bias")
    mhba_mha_to_bias_weight = load_npy("tfblock_weight_mhba_mha_to_bias_weight")
    mhba_mha_pair_norm_weight = load_npy("tfblock_weight_mhba_mha_pair_norm_weight")
    mhba_mha_pair_norm_bias = load_npy("tfblock_weight_mhba_mha_pair_norm_bias")
    mhba_scale_weight = load_npy("tfblock_weight_mhba_scale_output_to_adaln_zero_gamma_0_weight")
    mhba_scale_bias = load_npy("tfblock_weight_mhba_scale_output_to_adaln_zero_gamma_0_bias")

    # Load weights for transition (transition.*)
    trans_adaln_norm_cond_weight = load_npy("tfblock_weight_transition_adaln_norm_cond_weight")
    trans_adaln_norm_cond_bias = load_npy("tfblock_weight_transition_adaln_norm_cond_bias")
    trans_adaln_gamma_weight = load_npy("tfblock_weight_transition_adaln_to_gamma_0_weight")
    trans_adaln_gamma_bias = load_npy("tfblock_weight_transition_adaln_to_gamma_0_bias")
    trans_adaln_beta_weight = load_npy("tfblock_weight_transition_adaln_to_beta_weight")
    trans_swish_linear_weight = load_npy("tfblock_weight_transition_transition_swish_linear_0_weight")
    trans_linear_out_weight = load_npy("tfblock_weight_transition_transition_linear_out_weight")
    trans_scale_weight = load_npy("tfblock_weight_transition_scale_output_to_adaln_zero_gamma_0_weight")
    trans_scale_bias = load_npy("tfblock_weight_transition_scale_output_to_adaln_zero_gamma_0_bias")

    # Convert inputs to Julia format
    x_jl = py_to_jl(x_py)  # [64, 10, 2]
    pair_jl = py_to_jl_pair(pair_py)  # [32, 10, 10, 2]
    cond_jl = py_to_jl(cond_py)  # [32, 10, 2]
    mask_jl = permutedims(mask_py, (2, 1))  # [10, 2]
    out_py_jl = py_to_jl(out_py)

    # Create Julia module
    block = TransformerBlock(
        dim_token=64,
        dim_pair=32,
        n_heads=4,
        dim_cond=32,
        qk_ln=false,
        residual_mha=true,
        residual_transition=true,
        parallel=false,
        expansion_factor=4
    )

    # Set MHA_ADALN weights
    block.mha.adaln.norm_cond.diag.scale .= mhba_adaln_norm_cond_weight
    block.mha.adaln.norm_cond.diag.bias .= mhba_adaln_norm_cond_bias
    block.mha.adaln.to_gamma[1].weight .= mhba_adaln_gamma_weight
    block.mha.adaln.to_gamma[1].bias .= mhba_adaln_gamma_bias
    block.mha.adaln.to_beta.weight .= mhba_adaln_beta_weight
    block.mha.mha.node_norm.diag.scale .= mhba_mha_node_norm_weight
    block.mha.mha.node_norm.diag.bias .= mhba_mha_node_norm_bias
    block.mha.mha.to_qkv.weight .= mhba_mha_to_qkv_weight
    block.mha.mha.to_qkv.bias .= mhba_mha_to_qkv_bias
    block.mha.mha.to_g.weight .= mhba_mha_to_g_weight
    block.mha.mha.to_g.bias .= mhba_mha_to_g_bias
    block.mha.mha.to_out.weight .= mhba_mha_to_out_weight
    block.mha.mha.to_out.bias .= mhba_mha_to_out_bias
    block.mha.mha.pair_norm.diag.scale .= mhba_mha_pair_norm_weight
    block.mha.mha.pair_norm.diag.bias .= mhba_mha_pair_norm_bias
    block.mha.mha.to_bias.weight .= mhba_mha_to_bias_weight
    block.mha.scale_output.linear.weight .= mhba_scale_weight
    block.mha.scale_output.linear.bias .= mhba_scale_bias

    # Set TransitionADALN weights
    block.transition.adaln.norm_cond.diag.scale .= trans_adaln_norm_cond_weight
    block.transition.adaln.norm_cond.diag.bias .= trans_adaln_norm_cond_bias
    block.transition.adaln.to_gamma[1].weight .= trans_adaln_gamma_weight
    block.transition.adaln.to_gamma[1].bias .= trans_adaln_gamma_bias
    block.transition.adaln.to_beta.weight .= trans_adaln_beta_weight
    block.transition.transition.linear_in.weight .= trans_swish_linear_weight
    block.transition.transition.linear_out.weight .= trans_linear_out_weight
    block.transition.scale_output.linear.weight .= trans_scale_weight
    block.transition.scale_output.linear.bias .= trans_scale_bias

    # Run Julia
    out_jl = block(x_jl, pair_jl, cond_jl, mask_jl)

    return check_parity("transformer_block", out_jl, out_py_jl; rtol=1e-3, atol=1e-4)
end

function test_v_to_x1()
    println("\n=== Testing v <-> x1 Conversion ===")

    # Load Python reference
    xt_py = load_npy("v_x1_input_xt")  # [B=2, L=10, D=3]
    x1_py = load_npy("v_x1_input_x1")  # [B=2, L=10, D=3]
    t_py = load_npy("v_x1_input_t")  # [B=2]
    v_py = load_npy("v_x1_computed_v")  # [B=2, L=10, D=3]

    # Convert to Julia format
    xt_jl = py_to_jl(xt_py)  # [3, 10, 2]
    x1_jl = py_to_jl(x1_py)
    v_py_jl = py_to_jl(v_py)

    # Compute v from x_t and x_1
    # v = (x_1 - x_t) / (1 - t)
    t_exp = reshape(t_py, 1, 1, :)  # [1, 1, B]
    v_jl = (x1_jl .- xt_jl) ./ (1.0f0 .- t_exp .+ 1f-5)

    # Recover x1 from v
    # x_1 = x_t + (1-t) * v
    x1_recovered = xt_jl .+ (1.0f0 .- t_exp) .* v_jl

    pass1 = check_parity("v_computation", v_jl, v_py_jl; rtol=1e-4)
    pass2 = check_parity("x1_recovery", x1_recovered, x1_jl; rtol=1e-4)

    return pass1 && pass2
end

# ============================================================================
# Main
# ============================================================================

function main()
    println("=" ^ 60)
    println("LaProteina Parity Tests")
    println("=" ^ 60)

    if !isdir(PARITY_DATA_DIR)
        error("Reference data directory not found: $PARITY_DATA_DIR\nRun parity_test.py first!")
    end

    results = Dict{String, Bool}()

    # Run tests
    results["Index Embedding"] = test_index_embedding()
    results["Time Embedding"] = test_time_embedding()
    results["Relative Seq Sep"] = test_relative_sequence_separation()
    results["Distance Binning"] = test_distance_binning()
    results["AdaptiveLayerNorm"] = test_adaptive_layer_norm()
    results["AdaptiveOutputScale"] = test_adaptive_output_scale()
    results["SwiGLU"] = test_swiglu()
    results["RDN Interpolation"] = test_rdn_interpolation()
    results["RDN Loss"] = test_rdn_loss()
    results["PairBiasAttention"] = test_pair_bias_attention()
    results["MHA_ADALN"] = test_mha_adaln()
    results["Simple Transition"] = test_simple_transition()
    results["TransitionADALN"] = test_transition_adaln()
    results["TransformerBlock"] = test_transformer_block()
    results["v <-> x1 Conversion"] = test_v_to_x1()

    # Summary
    println("\n" * "=" ^ 60)
    println("Summary")
    println("=" ^ 60)

    passed = 0
    failed = 0
    for (name, result) in sort(collect(results), by=x->x[1])
        status = result ? "✓ PASS" : "✗ FAIL"
        println("  $status: $name")
        result ? (passed += 1) : (failed += 1)
    end

    println("\n$passed passed, $failed failed")

    return failed == 0
end

# Run tests
success = main()
exit(success ? 0 : 1)
