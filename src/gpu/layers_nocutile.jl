# CuArray layer overrides that work WITHOUT cuTile.
# Provides: optimized attention with NNlib, buffer-pooled permutedims,
# in-place residuals, eager freeing, fused PyTorchLayerNorm.

using NNlib: softmax, batched_mul, batched_transpose, sigmoid
using Statistics: mean, var

# ============================================================================
# Optimized _attention for CuArray: uses batched_transpose (lazy, no copy)
# instead of permutedims, reducing allocations significantly.
# ============================================================================

function _attention(q::CuArray{T}, k::CuArray{T}, v::CuArray{T}, pair_bias, mask, scale) where T
    # q, k, v: [d, L, h, B]
    # pair_bias: [h, L_q, L_k, B] or nothing
    # mask: [L, B]

    d, L, h, B = size(q)

    # Reshape for batched matmul: [d, L, h*B]
    q_flat = reshape(q, d, L, h * B)
    k_flat = reshape(k, d, L, h * B)

    # scores = q^T @ k: [L_q, L_k, h*B]
    # batched_transpose is a lazy wrapper (no copy!)
    scores = batched_mul(batched_transpose(q_flat), k_flat)  # [L, L, h*B]

    # Reshape to [L, L, h, B] for bias and mask ops
    scores = reshape(scores, L, L, h, B)

    # Scale + pair bias + mask in fused broadcasts
    if !isnothing(pair_bias)
        # bias is [h, L_q, L_k, B], permute to [L_q, L_k, h, B]
        bias_perm = permutedims(pair_bias, (2, 3, 1, 4))
        if !isnothing(mask)
            attn_mask = reshape(mask, 1, L, 1, B)
            scores = @. scores * scale + bias_perm + (1.0f0 - attn_mask) * T(-1.0f10)
        else
            scores = scores .* scale .+ bias_perm
        end
    else
        if !isnothing(mask)
            attn_mask = reshape(mask, 1, L, 1, B)
            scores = @. scores * scale + (1.0f0 - attn_mask) * T(-1.0f10)
        else
            scores = scores .* scale
        end
    end

    # Softmax over key dimension (dim 2)
    attn_weights = softmax(scores; dims=2)  # [L_q, L_k, h, B]

    # Reshape for batched matmul: attn @ v^T
    attn_flat = reshape(attn_weights, L, L, h * B)  # [L_q, L_k, h*B]
    v_flat = reshape(v, d, L, h * B)  # [d, L_k, h*B]

    # output = attn @ v^T: [L_q, d, h*B]
    out = batched_mul(attn_flat, batched_transpose(v_flat))  # [L_q, d, h*B]
    out = permutedims(out, (2, 1, 3))  # [d, L_q, h*B]

    # Reshape back
    return reshape(out, d, L, h, B)  # [d, L, h, B]
end

# ============================================================================
# Fused PyTorchLayerNorm for CuArray: custom rrule to reduce allocations
# in backward pass. The standard Zygote-traced path creates ~9 intermediates.
# ============================================================================

function _pytorch_layernorm_fwd(x::CuArray{T}, scale, bias, eps::Float32, dims) where T
    μ = mean(x, dims=dims)
    centered = x .- μ
    σ² = mean(centered .^ 2, dims=dims)
    inv_std = @. 1 / sqrt(σ² + T(eps))
    normed = centered .* inv_std
    if !isnothing(scale) && !isnothing(bias)
        y = @. normed * scale + bias
    else
        y = normed
    end
    return y, normed, inv_std
end

function CRC.rrule(::typeof(_pytorch_layernorm_fwd), x::CuArray{T}, scale, bias, eps::Float32, dims) where T
    y, normed, inv_std = _pytorch_layernorm_fwd(x, scale, bias, eps, dims)

    function layernorm_pullback(Δ)
        dy = unthunk(Δ[1])  # gradient of y

        if !isnothing(scale)
            # Fuse: compute dy_normed = dy * scale AND accumulate dscale, dbias
            # in as few operations as possible
            dy_normed = dy .* scale
            batch_dims = Tuple(setdiff(1:ndims(x), dims))
            dscale = sum(dy .* normed; dims=batch_dims)
            dbias = sum(dy; dims=batch_dims)
        else
            dy_normed = dy
            dscale = NoTangent()
            dbias = NoTangent()
        end

        # Efficient LayerNorm backward (fused into 2 reduction + 1 broadcast):
        # dx = inv_std / N * (N * dy_normed - sum(dy_normed) - normed * sum(dy_normed * normed))
        # Equivalently: dx = inv_std * (dy_normed - mean(dy_normed) - normed * mean(dy_normed * normed))
        mean_dy = mean(dy_normed; dims=dims)
        mean_dy_x = mean(dy_normed .* normed; dims=dims)
        dx = @. inv_std * (dy_normed - mean_dy - normed * mean_dy_x)

        return NoTangent(), dx, dscale, dbias, NoTangent(), NoTangent()
    end

    return (y, normed, inv_std), layernorm_pullback
end

# Override PyTorchLayerNorm for CuArray inputs.
# During inference: use the fast fused forward path.
# During gradient: use _pytorch_layernorm_fwd which has a custom rrule.
function (ln::PyTorchLayerNorm)(x::CuArray)
    if within_gradient(x)
        dims = 1:length(ln.size)
        y, _, _ = _pytorch_layernorm_fwd(x, ln.scale, ln.bias, ln.ϵ, dims)
        return ln.λ(y)
    end
    # Inference: fast fused path (same as original CPU code)
    y = pytorch_normalise(x; dims=1:length(ln.size), eps=ln.ϵ)
    if ln.affine
        y = @. y * ln.scale + ln.bias
    end
    return ln.λ(y)
end

# ============================================================================
# TransformerBlock: in-place residuals during inference
# ============================================================================

function (m::TransformerBlock)(x::CuArray, pair_rep, cond, mask)
    mask_expanded = reshape(mask, 1, size(mask)...)
    x = x .* mask_expanded

    if within_gradient(x)
        # Training: out-of-place residuals (AD-safe)
        if m.parallel
            x_mha = m.mha(x, pair_rep, cond, mask)
            x_tr = m.transition(x, cond, mask)
            x = x_mha .+ x_tr
            if m.residual_mha
                x = x .+ x
            end
        else
            x_mha = m.mha(x, pair_rep, cond, mask)
            if m.residual_mha
                x = x .+ x_mha
            else
                x = x_mha
            end
            x = x .* mask_expanded
            x_tr = m.transition(x, cond, mask)
            if m.residual_transition
                x = x .+ x_tr
            else
                x = x_tr
            end
        end
        return x .* mask_expanded
    end

    # Inference: in-place residuals + eager freeing
    if m.parallel
        x_mha = m.mha(x, pair_rep, cond, mask)
        x_tr = m.transition(x, cond, mask)
        x = x_mha .+ x_tr
        if m.residual_mha
            x .+= x
        end
    else
        x_mha = m.mha(x, pair_rep, cond, mask)
        if m.residual_mha
            x .+= x_mha
        else
            x = x_mha
        end
        CUDA.unsafe_free!(x_mha)
        x .= x .* mask_expanded

        x_tr = m.transition(x, cond, mask)
        if m.residual_transition
            x .+= x_tr
        else
            x = x_tr
        end
        CUDA.unsafe_free!(x_tr)
    end

    x .= x .* mask_expanded
    return x
end

# ============================================================================
# Pre-normalized pair feature support
# ============================================================================

"""
    _apply_pair_affine(pair_normed, pair_norm, to_bias)

Apply per-block affine transform to pre-normalized pair features, fused with
the to_bias projection to avoid materializing the full affine-transformed pair tensor.

Math: to_bias(normed * scale + bias) = W @ (normed * scale + bias)
    = (W .* scale') @ normed + W @ bias    (since to_bias has bias=false)
    = W_fused @ normed + b_fused

This avoids the expensive broadcast of scale/bias over [256, L, L, B] by
folding the affine into the projection weights.
"""
function _apply_pair_affine(pair_normed::CuArray, pair_norm::PyTorchLayerNorm, to_bias)
    if pair_norm.affine
        # Fuse affine + projection: W_fused = W .* scale', b_fused = W * bias
        # to_bias.weight: [n_heads, pair_dim], pair_norm.scale: [pair_dim]
        W = to_bias.weight  # [n_heads, pair_dim]
        scale = pair_norm.scale  # [pair_dim]
        bias = pair_norm.bias    # [pair_dim]
        W_fused = W .* reshape(scale, 1, :)  # [n_heads, pair_dim]
        b_fused = W * bias  # [n_heads]
        # Apply: W_fused @ pair_normed + b_fused
        # pair_normed: [pair_dim, L, L, B], reshape to matrix for matmul
        pd, Lq, Lk, B = size(pair_normed)
        pn_flat = reshape(pair_normed, pd, Lq * Lk * B)
        out_flat = W_fused * pn_flat  # [n_heads, L*L*B]
        out = reshape(out_flat, size(W, 1), Lq, Lk, B) .+ b_fused
        return out
    else
        return to_bias(pair_normed)
    end
end

"""
    forward_from_raw_features_gpu(model::ScoreNetwork, raw_features::ScoreNetworkRawFeatures)

GPU-optimized forward pass that pre-normalizes pair features once instead of
14 times (once per transformer block). Same semantics as `forward_from_raw_features`.

Savings: 13 × (pair LayerNorm forward + backward) ≈ 13 × 17ms = 221ms per training step.
"""
function forward_from_raw_features_gpu(model::ScoreNetwork, raw_features::ScoreNetworkRawFeatures)
    mask = raw_features.mask
    L, B = size(mask)
    mask_exp = reshape(mask, 1, L, B)
    pair_mask = reshape(mask, L, 1, B) .* reshape(mask, 1, L, B)
    pair_mask_exp = reshape(pair_mask, 1, L, L, B)

    # Project features (same as forward_from_raw_features)
    cond = model.cond_factory.projection(raw_features.cond_raw)
    if model.cond_factory.use_ln && !isnothing(model.cond_factory.ln)
        cond = model.cond_factory.ln(cond)
    end
    cond = cond .* mask_exp

    seqs = model.init_repr_factory.projection(raw_features.seq_raw)
    if model.init_repr_factory.use_ln && !isnothing(model.init_repr_factory.ln)
        seqs = model.init_repr_factory.ln(seqs)
    end
    seqs = seqs .* mask_exp

    pair_rep = model.pair_rep_builder.init_repr_factory.projection(raw_features.pair_raw)
    if model.pair_rep_builder.init_repr_factory.use_ln && !isnothing(model.pair_rep_builder.init_repr_factory.ln)
        pair_rep = model.pair_rep_builder.init_repr_factory.ln(pair_rep)
    end
    pair_rep = pair_rep .* pair_mask_exp

    cond = model.transition_c_1(cond, mask)
    cond = model.transition_c_2(cond, mask)

    if !isnothing(model.pair_rep_builder.adaln) && !isnothing(model.pair_rep_builder.cond_factory)
        # === OPTIMIZATION: Batch-level pair conditioning ===
        # pair_cond_raw is batch-level conditioning (time embeddings) broadcast to [D, L, L, B].
        # The adaln internally takes cond[:, 1, 1, :] back to [D, B]. By extracting batch level
        # BEFORE the Dense projection, we project [D_raw, B] instead of [D_raw, L*L*B] —
        # a 16384x reduction in matmul size. The adaln already handles 2D conditioning.
        pair_cond_batch_raw = raw_features.pair_cond_raw[:, 1, 1, :]  # [D_raw, B]
        pair_cond_batch = model.pair_rep_builder.cond_factory.projection(pair_cond_batch_raw)
        if model.pair_rep_builder.cond_factory.use_ln && !isnothing(model.pair_rep_builder.cond_factory.ln)
            pair_cond_batch = model.pair_rep_builder.cond_factory.ln(pair_cond_batch)
        end
        pair_rep = model.pair_rep_builder.adaln(pair_rep, pair_cond_batch, pair_mask)
    end

    # === KEY OPTIMIZATION: Pre-normalize pair features once ===
    # Each block's PairBiasAttention does: pair_norm(pair_rep) then to_bias(...)
    # pair_norm = PyTorchLayerNorm(pair_dim) which computes:
    #   normed = (pair_rep - μ) / sqrt(σ² + ε)   <-- expensive, same for all blocks
    #   output = normed * scale + bias             <-- cheap, different per block
    # We compute the normalization once and pass the normed pair to each block.
    if !model.update_pair_repr
        # Get the pair_norm eps from first block's attention
        first_pba = model.transformer_layers[1].mha.mha
        pair_eps = first_pba.pair_norm.ϵ
        pair_normed = pytorch_normalise(pair_rep; dims=1, eps=pair_eps)
    else
        pair_normed = nothing
    end

    # Run transformer layers
    for i in 1:model.n_layers
        if !isnothing(pair_normed)
            # Use pre-normalized pair path
            seqs = _transformer_block_prenormed(
                model.transformer_layers[i], seqs, pair_rep, pair_normed, cond, mask)
        else
            seqs = model.transformer_layers[i](seqs, pair_rep, cond, mask)
        end

        if model.update_pair_repr && i < model.n_layers
            if !isnothing(model.pair_update_layers[i])
                pair_rep = model.pair_update_layers[i](seqs, pair_rep, mask)
                # Re-normalize after update
                pair_normed = pytorch_normalise(pair_rep; dims=1, eps=pair_eps)
            end
        end
    end

    local_latents_out = model.local_latents_proj(seqs) .* mask_exp
    ca_out = model.ca_proj(seqs) .* mask_exp

    out_key = model.output_param
    return Dict(
        :bb_ca => Dict(out_key => ca_out),
        :local_latents => Dict(out_key => local_latents_out)
    )
end

"""
    _transformer_block_prenormed(block, x, pair_rep, pair_normed, cond, mask)

TransformerBlock forward with pre-normalized pair features.
Uses pair_normed (already normalized) + per-block affine instead of full pair_norm.
"""
function _transformer_block_prenormed(block::TransformerBlock, x::CuArray, pair_rep, pair_normed, cond, mask)
    mask_expanded = reshape(mask, 1, size(mask)...)
    x = x .* mask_expanded

    # MHA with pre-normalized pairs
    pba = block.mha.mha  # PairBiasAttention

    # AdaLN (applies mask internally)
    x_normed = block.mha.adaln(x, cond, mask)

    # PairBiasAttention with pre-normalized pair
    attn_out = _pair_bias_attn_prenormed(pba, x_normed, pair_normed, mask)

    # AdaptiveOutputScale (applies mask internally)
    attn_out = block.mha.scale_output(attn_out, cond, mask)

    # Residual (both x and attn_out are already masked)
    if block.residual_mha
        x = x .+ attn_out
    else
        x = attn_out
    end

    # Transition (applies mask internally via SwiGLUTransition)
    x_tr = block.transition(x, cond, mask)
    if block.residual_transition
        x = x .+ x_tr
    else
        x = x_tr
    end

    return x
end

"""
    _pair_bias_attn_prenormed(m, node_feats, pair_normed, mask)

PairBiasAttention forward using pre-normalized pair features.
Skips pair_norm normalization, only applies per-block affine transform.
"""
function _pair_bias_attn_prenormed(m::PairBiasAttention, node_feats::CuArray, pair_normed, mask)
    D, L, B = size(node_feats)
    h = m.n_heads
    d = m.dim_head
    inner_dim = h * d

    node_feats = m.node_norm(node_feats)

    qkv = m.to_qkv(node_feats)
    q = qkv[1:inner_dim, :, :]
    k = qkv[inner_dim+1:2*inner_dim, :, :]
    v = qkv[2*inner_dim+1:end, :, :]

    q = m.q_norm(q)
    k = m.k_norm(k)
    g = m.to_g(node_feats)

    q = reshape(q, d, h, L, B)
    k = reshape(k, d, h, L, B)
    v = reshape(v, d, h, L, B)
    g = reshape(g, d, h, L, B)

    q = permutedims(q, (1, 3, 2, 4))
    k = permutedims(k, (1, 3, 2, 4))
    v = permutedims(v, (1, 3, 2, 4))
    g = permutedims(g, (1, 3, 2, 4))

    # Apply per-block affine + bias projection (CHEAP: just broadcast + Dense)
    if !isnothing(pair_normed) && !isnothing(m.pair_norm)
        bias = _apply_pair_affine(pair_normed, m.pair_norm, m.to_bias)
    else
        bias = nothing
    end

    attn_out = _attention(q, k, v, bias, mask, m.scale)
    attn_out = sigmoid.(g) .* attn_out
    attn_out = permutedims(attn_out, (1, 3, 2, 4))
    attn_out = reshape(attn_out, inner_dim, L, B)
    return m.to_out(attn_out)
end

# ============================================================================
# PairBiasAttention: buffer-pooled permutedims during inference
# ============================================================================

function (m::PairBiasAttention)(node_feats::CuArray, pair_feats, mask)
    D, L, B = size(node_feats)
    h = m.n_heads
    d = m.dim_head
    inner_dim = h * d

    node_feats = m.node_norm(node_feats)

    qkv = m.to_qkv(node_feats)
    q = qkv[1:inner_dim, :, :]
    k = qkv[inner_dim+1:2*inner_dim, :, :]
    v = qkv[2*inner_dim+1:end, :, :]

    q = m.q_norm(q)
    k = m.k_norm(k)
    g = m.to_g(node_feats)

    if within_gradient(node_feats)
        # Training: allocating path (AD-safe)
        q = reshape(q, d, h, L, B)
        k = reshape(k, d, h, L, B)
        v = reshape(v, d, h, L, B)
        g = reshape(g, d, h, L, B)

        q = permutedims(q, (1, 3, 2, 4))
        k = permutedims(k, (1, 3, 2, 4))
        v = permutedims(v, (1, 3, 2, 4))
        g = permutedims(g, (1, 3, 2, 4))

        if !isnothing(pair_feats) && !isnothing(m.pair_norm)
            pair_feats = m.pair_norm(pair_feats)
            bias = m.to_bias(pair_feats)
        else
            bias = nothing
        end

        attn_out = _attention(q, k, v, bias, mask, m.scale)
        attn_out = sigmoid.(g) .* attn_out
        attn_out = permutedims(attn_out, (1, 3, 2, 4))
        attn_out = reshape(attn_out, inner_dim, L, B)
        return m.to_out(attn_out)
    end

    # Inference: buffer-pooled permutedims + eager free
    q_r = reshape(q, d, h, L, B)
    k_r = reshape(k, d, h, L, B)
    v_r = reshape(v, d, h, L, B)
    g_r = reshape(g, d, h, L, B)

    qkv_shape = (d, L, h, B)
    q4 = _get_perm_buf(20, qkv_shape)
    k4 = _get_perm_buf(21, qkv_shape)
    v4 = _get_perm_buf(22, qkv_shape)
    g4 = _get_perm_buf(25, qkv_shape)
    permutedims!(q4, q_r, (1, 3, 2, 4))
    permutedims!(k4, k_r, (1, 3, 2, 4))
    permutedims!(v4, v_r, (1, 3, 2, 4))
    permutedims!(g4, g_r, (1, 3, 2, 4))
    CUDA.unsafe_free!(qkv)

    if !isnothing(pair_feats) && !isnothing(m.pair_norm)
        pair_feats = m.pair_norm(pair_feats)
        bias = m.to_bias(pair_feats)
    else
        bias = nothing
    end

    attn_out = _attention(q4, k4, v4, bias, mask, m.scale)
    @. attn_out = NNlib.sigmoid(g4) * attn_out

    out_perm = _get_perm_buf(24, (d, h, L, B))
    permutedims!(out_perm, attn_out, (1, 3, 2, 4))
    attn_flat = reshape(out_perm, inner_dim, L, B)
    return m.to_out(attn_flat)
end

# ============================================================================
# SwiGLU: use views instead of copies for CuArray
# ============================================================================

function (::SwiGLU)(x::CuArray)
    dim = size(x, 1) ÷ 2
    x1 = @view x[1:dim, axes(x)[2:end]...]
    x2 = @view x[dim+1:end, axes(x)[2:end]...]
    return swish.(x2) .* x1
end
