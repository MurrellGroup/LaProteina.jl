# CuArray-specialized forward methods for LaProteina layers (cuTile path).
# Uses cuTile fused kernels (LayerNorm, Flash Attention), buffer pooling,
# in-place ops, eager freeing, pre-normalized pair features, and batch-level
# pair conditioning.

using NNlib: softmax, batched_mul, batched_transpose, sigmoid

# ============================================================================
# PyTorchLayerNorm: fused cuTile kernel via dispatch hooks
# ============================================================================

function (ln::PyTorchLayerNorm)(x::CuArray)
    if ln.affine
        y = layernorm_forward(x, ln.scale, ln.bias, Float32(ln.ϵ))
    else
        y = layernorm_noaffine_forward(x, Float32(ln.ϵ))
    end
    return ln.λ(y)
end

# ============================================================================
# Flash attention helper: combines bias + mask into FMHA bias tensor
# ============================================================================

"""
    _flash_attn(q4, k4, v4, bias_raw, mask, scale, L, h, B)

Run flash attention with pair bias and mask.
q4, k4, v4: [d, L, h, B] (FMHA layout)
bias_raw: [h, L_q, L_k, B] or nothing (LaProteina layout)
mask: [L, B] or nothing
"""
function _flash_attn(q4::CuArray{T}, k4::CuArray{T}, v4::CuArray{T},
                     bias_raw, mask, scale, L::Int, h::Int, B::Int) where T
    if !isnothing(bias_raw)
        # FMHA expects bias: (SeqK, SeqQ, H, B) = permutedims(bias, (3, 2, 1, 4))
        bias4 = permutedims(bias_raw, (3, 2, 1, 4))  # [L_k, L_q, h, B]

        # Apply mask as additive bias: mask [L, B] -> [L_k, 1, 1, B]
        if !isnothing(mask)
            mask_bias = reshape((1.0f0 .- mask) .* (-1.0f10), L, 1, 1, B)
            bias4 = bias4 .+ mask_bias
        end

        return flash_attention_bias_forward(q4, k4, v4, bias4; scale=scale)
    else
        # No pair bias — apply mask as bias if needed
        if !isnothing(mask)
            mask_bias = reshape((1.0f0 .- mask) .* (-1.0f10), L, 1, 1, B)
            mask_bias_full = repeat(mask_bias, 1, L, h, 1)
            return flash_attention_bias_forward(q4, k4, v4, mask_bias_full; scale=scale)
        else
            return flash_attention_forward(q4, k4, v4; scale=scale)
        end
    end
end

# ============================================================================
# PairBiasAttention: flash attention with pair bias
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
        # Training path: allocating permutedims, dispatch hooks with rrules
        q = reshape(q, d, h, L, B)
        k = reshape(k, d, h, L, B)
        v = reshape(v, d, h, L, B)

        q4 = permutedims(q, (1, 3, 2, 4))
        k4 = permutedims(k, (1, 3, 2, 4))
        v4 = permutedims(v, (1, 3, 2, 4))

        # Compute pair bias
        if !isnothing(pair_feats) && !isnothing(m.pair_norm)
            pair_normed = m.pair_norm(pair_feats)
            bias_raw = m.to_bias(pair_normed)  # [h, L, L, B]
        else
            bias_raw = nothing
        end

        attn_out = _flash_attn(q4, k4, v4, bias_raw, mask, m.scale, L, h, B)

        # Apply sigmoid gating in [d, L, h, B] space (matching flash attn output)
        g = reshape(g, d, h, L, B)
        g = permutedims(g, (1, 3, 2, 4))  # [d, L, h, B]
        attn_out = sigmoid.(g) .* attn_out

        # Permute back: [d, L, h, B] -> [d, h, L, B] -> [inner_dim, L, B]
        attn_out = permutedims(attn_out, (1, 3, 2, 4))
        attn_out = reshape(attn_out, inner_dim, L, B)

        return m.to_out(attn_out)
    end

    # Inference path: buffer-pooled permutedims, eager freeing
    q_r = reshape(q, d, h, L, B)
    k_r = reshape(k, d, h, L, B)
    v_r = reshape(v, d, h, L, B)

    qkv_shape = (d, L, h, B)
    q4 = _get_perm_buf(20, qkv_shape)
    k4 = _get_perm_buf(21, qkv_shape)
    v4 = _get_perm_buf(22, qkv_shape)
    permutedims!(q4, q_r, (1, 3, 2, 4))
    permutedims!(k4, k_r, (1, 3, 2, 4))
    permutedims!(v4, v_r, (1, 3, 2, 4))
    CUDA.unsafe_free!(qkv)

    # Compute pair bias
    if !isnothing(pair_feats) && !isnothing(m.pair_norm)
        pair_normed = m.pair_norm(pair_feats)
        bias_raw = m.to_bias(pair_normed)  # [h, L, L, B]
        bias4 = permutedims(bias_raw, (3, 2, 1, 4))  # [L_k, L_q, h, B]
        CUDA.unsafe_free!(bias_raw)

        if !isnothing(mask)
            mask_bias = reshape((1.0f0 .- mask) .* (-1.0f10), L, 1, 1, B)
            bias4 = bias4 .+ mask_bias
        end

        flash_out = _get_perm_buf(23, qkv_shape)
        flash_attention_bias(q4, k4, v4, bias4; scale=m.scale, output=flash_out)
        CUDA.unsafe_free!(bias4)
    else
        if !isnothing(mask)
            mask_bias = reshape((1.0f0 .- mask) .* (-1.0f10), L, 1, 1, B)
            mask_bias_full = repeat(mask_bias, 1, L, h, 1)
            flash_out = _get_perm_buf(23, qkv_shape)
            flash_attention_bias(q4, k4, v4, mask_bias_full; scale=m.scale, output=flash_out)
        else
            flash_out = _get_perm_buf(23, qkv_shape)
            flash_attention(q4, k4, v4; scale=m.scale, output=flash_out)
        end
    end

    # Apply sigmoid gating in [d, L, h, B] space (matching flash attn output)
    g_r = reshape(g, d, h, L, B)
    g_perm = _get_perm_buf(25, qkv_shape)
    permutedims!(g_perm, g_r, (1, 3, 2, 4))  # [d, L, h, B]
    @. flash_out = NNlib.sigmoid(g_perm) * flash_out

    # Permute back: [d, L, h, B] -> [d, h, L, B] -> [inner_dim, L, B]
    out_perm = _get_perm_buf(24, (d, h, L, B))
    permutedims!(out_perm, flash_out, (1, 3, 2, 4))
    attn_out = reshape(out_perm, inner_dim, L, B)

    return m.to_out(attn_out)
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
"""
function _apply_pair_affine(pair_normed::CuArray, pair_norm::PyTorchLayerNorm, to_bias)
    if pair_norm.affine
        W = to_bias.weight  # [n_heads, pair_dim]
        scale = pair_norm.scale  # [pair_dim]
        bias = pair_norm.bias    # [pair_dim]
        W_fused = W .* reshape(scale, 1, :)  # [n_heads, pair_dim]
        b_fused = W * bias  # [n_heads]
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
    _pair_bias_attn_prenormed(m, node_feats, pair_normed, mask)

PairBiasAttention forward using pre-normalized pair features + flash attention.
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
        bias_raw = _apply_pair_affine(pair_normed, m.pair_norm, m.to_bias)
    else
        bias_raw = nothing
    end

    attn_out = _flash_attn(q, k, v, bias_raw, mask, m.scale, L, h, B)

    attn_out = sigmoid.(g) .* attn_out
    attn_out = permutedims(attn_out, (1, 3, 2, 4))
    attn_out = reshape(attn_out, inner_dim, L, B)
    return m.to_out(attn_out)
end

# ============================================================================
# TransformerBlock: pre-normalized pair path + in-place residuals
# ============================================================================

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

    # PairBiasAttention with pre-normalized pair + flash attention
    attn_out = _pair_bias_attn_prenormed(pba, x_normed, pair_normed, mask)

    # AdaptiveOutputScale (applies mask internally)
    attn_out = block.mha.scale_output(attn_out, cond, mask)

    # Residual
    if block.residual_mha
        x = x .+ attn_out
    else
        x = attn_out
    end

    # Transition
    x_tr = block.transition(x, cond, mask)
    if block.residual_transition
        x = x .+ x_tr
    else
        x = x_tr
    end

    return x
end

function (m::TransformerBlock)(x::CuArray, pair_rep, cond, mask)
    mask_expanded = reshape(mask, 1, size(mask)...)
    x = x .* mask_expanded

    if within_gradient(x)
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
# SwiGLU: use views instead of copies for CuArray
# ============================================================================

function (::SwiGLU)(x::CuArray)
    dim = size(x, 1) ÷ 2
    x1 = @view x[1:dim, axes(x)[2:end]...]
    x2 = @view x[dim+1:end, axes(x)[2:end]...]
    return swish.(x2) .* x1
end

# ============================================================================
# GPU-optimized forward pass with all optimizations
# ============================================================================

"""
    forward_from_raw_features_gpu(model::ScoreNetwork, raw_features::ScoreNetworkRawFeatures)

GPU-optimized forward pass that combines:
1. Pre-normalized pair features (normalize once, apply per-block affine)
2. Batch-level pair conditioning (project [D,B] instead of [D,L*L*B])
3. Flash attention via cuTile kernels
4. Fused LayerNorm via cuTile kernels
"""
function forward_from_raw_features_gpu(model::ScoreNetwork, raw_features::ScoreNetworkRawFeatures)
    mask = raw_features.mask
    L, B = size(mask)
    mask_exp = reshape(mask, 1, L, B)
    pair_mask = reshape(mask, L, 1, B) .* reshape(mask, 1, L, B)
    pair_mask_exp = reshape(pair_mask, 1, L, L, B)

    # Project features
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
        pair_cond_batch_raw = raw_features.pair_cond_raw[:, 1, 1, :]  # [D_raw, B]
        pair_cond_batch = model.pair_rep_builder.cond_factory.projection(pair_cond_batch_raw)
        if model.pair_rep_builder.cond_factory.use_ln && !isnothing(model.pair_rep_builder.cond_factory.ln)
            pair_cond_batch = model.pair_rep_builder.cond_factory.ln(pair_cond_batch)
        end
        pair_rep = model.pair_rep_builder.adaln(pair_rep, pair_cond_batch, pair_mask)
    end

    # === KEY OPTIMIZATION: Pre-normalize pair features once ===
    if !model.update_pair_repr
        first_pba = model.transformer_layers[1].mha.mha
        pair_eps = first_pba.pair_norm.ϵ
        pair_normed = pytorch_normalise(pair_rep; dims=1, eps=pair_eps)
    else
        pair_normed = nothing
    end

    # Run transformer layers
    for i in 1:model.n_layers
        if !isnothing(pair_normed)
            seqs = _transformer_block_prenormed(
                model.transformer_layers[i], seqs, pair_rep, pair_normed, cond, mask)
        else
            seqs = model.transformer_layers[i](seqs, pair_rep, cond, mask)
        end

        if model.update_pair_repr && i < model.n_layers
            if !isnothing(model.pair_update_layers[i])
                pair_rep = model.pair_update_layers[i](seqs, pair_rep, mask)
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
