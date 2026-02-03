# Weight loading utilities for pretrained models

using NPZ
using Flux

# Helper functions
_get_w(weights::Dict, key::String) = haskey(weights, key) ? Float32.(weights[key]) : nothing
_get_w(weights::Dict, prefix::String, key::String) = _get_w(weights, prefix * key)

"""
    load_score_network_weights!(model::ScoreNetwork, weights_path::String)

Load pretrained weights into ScoreNetwork from NPZ file.
"""
function load_score_network_weights!(model::ScoreNetwork, weights_path::String)
    weights = npzread(weights_path)

    # init_repr_factory: sequence feature projection
    w = _get_w(weights, "init_repr_factory.linear_out.weight")
    if !isnothing(w)
        model.init_repr_factory.projection.weight .= w
    end

    # cond_factory: conditioning projection
    w = _get_w(weights, "cond_factory.linear_out.weight")
    if !isnothing(w)
        model.cond_factory.projection.weight .= w
    end

    # pair_repr_builder
    _load_pair_repr_builder!(model.pair_rep_builder, weights, "pair_repr_builder.")

    # transition_c_1
    _load_conditioning_transition!(model.transition_c_1, weights, "transition_c_1.")

    # transition_c_2
    _load_conditioning_transition!(model.transition_c_2, weights, "transition_c_2.")

    # transformer_layers
    for (i, layer) in enumerate(model.transformer_layers)
        prefix = "transformer_layers.$(i-1)."
        _load_transformer_block!(layer, weights, prefix)
    end

    # Output projections
    # local_latents_proj: LayerNorm + Dense
    w = _get_w(weights, "local_latents_linear.0.weight")
    b = _get_w(weights, "local_latents_linear.0.bias")
    if !isnothing(w) && !isnothing(b)
        model.local_latents_proj.layers[1].scale .= w
        model.local_latents_proj.layers[1].bias .= b
    end
    w = _get_w(weights, "local_latents_linear.1.weight")
    if !isnothing(w)
        model.local_latents_proj.layers[2].weight .= w
    end

    # ca_proj: LayerNorm + Dense
    w = _get_w(weights, "ca_linear.0.weight")
    b = _get_w(weights, "ca_linear.0.bias")
    if !isnothing(w) && !isnothing(b)
        model.ca_proj.layers[1].scale .= w
        model.ca_proj.layers[1].bias .= b
    end
    w = _get_w(weights, "ca_linear.1.weight")
    if !isnothing(w)
        model.ca_proj.layers[2].weight .= w
    end

    return model
end

function _load_pair_repr_builder!(builder::PairReprBuilder, weights::Dict, prefix::String)
    # init_repr_factory
    w = _get_w(weights, prefix, "init_repr_factory.linear_out.weight")
    if !isnothing(w)
        builder.init_repr_factory.projection.weight .= w
    end

    # LayerNorm if present
    if builder.init_repr_factory.use_ln && !isnothing(builder.init_repr_factory.ln)
        w = _get_w(weights, prefix, "init_repr_factory.ln_out.weight")
        b = _get_w(weights, prefix, "init_repr_factory.ln_out.bias")
        if !isnothing(w) && !isnothing(b)
            builder.init_repr_factory.ln.scale .= w
            builder.init_repr_factory.ln.bias .= b
        end
    end

    # cond_factory
    if !isnothing(builder.cond_factory)
        w = _get_w(weights, prefix, "cond_factory.linear_out.weight")
        if !isnothing(w)
            builder.cond_factory.projection.weight .= w
        end
        if builder.cond_factory.use_ln && !isnothing(builder.cond_factory.ln)
            w = _get_w(weights, prefix, "cond_factory.ln_out.weight")
            b = _get_w(weights, prefix, "cond_factory.ln_out.bias")
            if !isnothing(w) && !isnothing(b)
                builder.cond_factory.ln.scale .= w
                builder.cond_factory.ln.bias .= b
            end
        end
    end

    # adaln
    if !isnothing(builder.adaln)
        _load_adaln_identical!(builder.adaln, weights, prefix * "adaln.")
    end
end

function _load_adaln_identical!(adaln::AdaptiveLayerNormIdentical, weights::Dict, prefix::String)
    # norm_cond
    if !isnothing(adaln.norm_cond)
        w = _get_w(weights, prefix, "norm_cond.weight")
        b = _get_w(weights, prefix, "norm_cond.bias")
        if !isnothing(w) && !isnothing(b)
            adaln.norm_cond.scale .= w
            adaln.norm_cond.bias .= b
        end
    end

    # to_gamma: Dense + sigmoid (in Chain)
    w = _get_w(weights, prefix, "to_gamma.0.weight")
    b = _get_w(weights, prefix, "to_gamma.0.bias")
    if !isnothing(w)
        adaln.to_gamma.layers[1].weight .= w
        if !isnothing(b)
            adaln.to_gamma.layers[1].bias .= b
        end
    end

    # to_beta
    w = _get_w(weights, prefix, "to_beta.weight")
    if !isnothing(w)
        adaln.to_beta.weight .= w
    end
end

function _load_conditioning_transition!(trans::ConditioningTransition, weights::Dict, prefix::String)
    # swish_linear.0 -> linear_in
    w = _get_w(weights, prefix, "swish_linear.0.weight")
    if !isnothing(w)
        trans.transition.linear_in.weight .= w
    end

    # linear_out
    w = _get_w(weights, prefix, "linear_out.weight")
    if !isnothing(w)
        trans.transition.linear_out.weight .= w
    end
end

function _load_transformer_block!(block::TransformerBlock, weights::Dict, prefix::String)
    # mhba (MultiHeadBiasedAttentionADALN) - stored in block.mha
    mhba_prefix = prefix * "mhba."

    # adaln
    _load_adaln!(block.mha.adaln, weights, mhba_prefix * "adaln.")

    # mha (PairBiasAttention)
    mha_prefix = mhba_prefix * "mha."
    _load_pair_bias_attention!(block.mha.mha, weights, mha_prefix)

    # scale_output
    _load_adaptive_output_scale!(block.mha.scale_output, weights, mhba_prefix * "scale_output.")

    # transition (TransitionADALN)
    trans_prefix = prefix * "transition."

    # adaln
    _load_adaln!(block.transition.adaln, weights, trans_prefix * "adaln.")

    # transition (SwiGLUTransition)
    swiglu_prefix = trans_prefix * "transition."
    w = _get_w(weights, swiglu_prefix, "swish_linear.0.weight")
    if !isnothing(w)
        block.transition.transition.linear_in.weight .= w
    end
    w = _get_w(weights, swiglu_prefix, "linear_out.weight")
    if !isnothing(w)
        block.transition.transition.linear_out.weight .= w
    end

    # scale_output
    _load_adaptive_output_scale!(block.transition.scale_output, weights, trans_prefix * "scale_output.")
end

function _load_adaln!(adaln::ProteINAAdaLN, weights::Dict, prefix::String)
    # norm_cond
    w = _get_w(weights, prefix, "norm_cond.weight")
    b = _get_w(weights, prefix, "norm_cond.bias")
    if !isnothing(w) && !isnothing(b)
        adaln.norm_cond.scale .= w
        adaln.norm_cond.bias .= b
    end

    # to_gamma
    w = _get_w(weights, prefix, "to_gamma.0.weight")
    b = _get_w(weights, prefix, "to_gamma.0.bias")
    if !isnothing(w)
        adaln.to_gamma.layers[1].weight .= w
        if !isnothing(b)
            adaln.to_gamma.layers[1].bias .= b
        end
    end

    # to_beta
    w = _get_w(weights, prefix, "to_beta.weight")
    if !isnothing(w)
        adaln.to_beta.weight .= w
    end
end

function _load_pair_bias_attention!(mha::PairBiasAttention, weights::Dict, prefix::String)
    # node_norm
    w = _get_w(weights, prefix, "node_norm.weight")
    b = _get_w(weights, prefix, "node_norm.bias")
    if !isnothing(w) && !isnothing(b)
        mha.node_norm.scale .= w
        mha.node_norm.bias .= b
    end

    # pair_norm
    w = _get_w(weights, prefix, "pair_norm.weight")
    b = _get_w(weights, prefix, "pair_norm.bias")
    if !isnothing(w) && !isnothing(b)
        mha.pair_norm.scale .= w
        mha.pair_norm.bias .= b
    end

    # q_layer_norm, k_layer_norm (if present)
    if !isnothing(mha.q_norm)
        w = _get_w(weights, prefix, "q_layer_norm.weight")
        b = _get_w(weights, prefix, "q_layer_norm.bias")
        if !isnothing(w) && !isnothing(b)
            mha.q_norm.scale .= w
            mha.q_norm.bias .= b
        end
    end
    if !isnothing(mha.k_norm)
        w = _get_w(weights, prefix, "k_layer_norm.weight")
        b = _get_w(weights, prefix, "k_layer_norm.bias")
        if !isnothing(w) && !isnothing(b)
            mha.k_norm.scale .= w
            mha.k_norm.bias .= b
        end
    end

    # to_qkv
    w = _get_w(weights, prefix, "to_qkv.weight")
    b = _get_w(weights, prefix, "to_qkv.bias")
    if !isnothing(w)
        mha.to_qkv.weight .= w
        if !isnothing(b)
            mha.to_qkv.bias .= b
        end
    end

    # to_g
    w = _get_w(weights, prefix, "to_g.weight")
    b = _get_w(weights, prefix, "to_g.bias")
    if !isnothing(w)
        mha.to_g.weight .= w
        if !isnothing(b)
            mha.to_g.bias .= b
        end
    end

    # to_out (called to_out_node in Python)
    w = _get_w(weights, prefix, "to_out_node.weight")
    b = _get_w(weights, prefix, "to_out_node.bias")
    if !isnothing(w)
        mha.to_out.weight .= w
        if !isnothing(b)
            mha.to_out.bias .= b
        end
    end

    # to_bias
    w = _get_w(weights, prefix, "to_bias.weight")
    if !isnothing(w)
        mha.to_bias.weight .= w
    end
end

function _load_adaptive_output_scale!(aos::AdaptiveOutputScale, weights::Dict, prefix::String)
    # linear: to_adaln_zero_gamma.0
    w = _get_w(weights, prefix, "to_adaln_zero_gamma.0.weight")
    b = _get_w(weights, prefix, "to_adaln_zero_gamma.0.bias")
    if !isnothing(w)
        aos.linear.weight .= w
        if !isnothing(b)
            aos.linear.bias .= b
        end
    end
end

"""
    load_decoder_weights!(model::DecoderTransformer, weights_path::String)

Load pretrained weights into DecoderTransformer from NPZ file.
"""
function load_decoder_weights!(model::DecoderTransformer, weights_path::String)
    weights = npzread(weights_path)

    # init_repr_factory
    w = _get_w(weights, "init_repr_factory.linear_out.weight")
    if !isnothing(w)
        model.init_repr_factory.projection.weight .= w
    end

    # pair_rep_factory
    w = _get_w(weights, "pair_rep_factory.linear_out.weight")
    if !isnothing(w)
        model.pair_rep_factory.projection.weight .= w
    end

    # transition_c_1, transition_c_2
    _load_conditioning_transition!(model.transition_c_1, weights, "transition_c_1.")
    _load_conditioning_transition!(model.transition_c_2, weights, "transition_c_2.")

    # transformer_layers
    for (i, layer) in enumerate(model.transformer_layers)
        prefix = "transformer_layers.$(i-1)."
        _load_transformer_block!(layer, weights, prefix)
    end

    # Output projections
    # logit_proj: LayerNorm + Dense
    w = _get_w(weights, "logit_linear.0.weight")
    b = _get_w(weights, "logit_linear.0.bias")
    if !isnothing(w) && !isnothing(b)
        model.logit_proj.layers[1].scale .= w
        model.logit_proj.layers[1].bias .= b
    end
    w = _get_w(weights, "logit_linear.1.weight")
    if !isnothing(w)
        model.logit_proj.layers[2].weight .= w
    end

    # struct_proj: LayerNorm + Dense
    w = _get_w(weights, "struct_linear.0.weight")
    b = _get_w(weights, "struct_linear.0.bias")
    if !isnothing(w) && !isnothing(b)
        model.struct_proj.layers[1].scale .= w
        model.struct_proj.layers[1].bias .= b
    end
    w = _get_w(weights, "struct_linear.1.weight")
    if !isnothing(w)
        model.struct_proj.layers[2].weight .= w
    end

    return model
end

"""
    load_encoder_weights!(model::EncoderTransformer, weights_path::String)

Load pretrained weights into EncoderTransformer from NPZ file.
"""
function load_encoder_weights!(model::EncoderTransformer, weights_path::String)
    weights = npzread(weights_path)

    # init_repr_factory
    w = _get_w(weights, "init_repr_factory.linear_out.weight")
    if !isnothing(w)
        model.init_repr_factory.projection.weight .= w
    end

    # pair_rep_factory
    w = _get_w(weights, "pair_rep_factory.linear_out.weight")
    if !isnothing(w)
        model.pair_rep_factory.projection.weight .= w
    end

    # transition_c_1, transition_c_2
    _load_conditioning_transition!(model.transition_c_1, weights, "transition_c_1.")
    _load_conditioning_transition!(model.transition_c_2, weights, "transition_c_2.")

    # transformer_layers
    for (i, layer) in enumerate(model.transformer_layers)
        prefix = "transformer_layers.$(i-1)."
        _load_transformer_block!(layer, weights, prefix)
    end

    # latent_proj: LayerNorm + Dense (called latent_decoder_mean_n_log_scale in Python)
    w = _get_w(weights, "latent_decoder_mean_n_log_scale.0.weight")
    b = _get_w(weights, "latent_decoder_mean_n_log_scale.0.bias")
    if !isnothing(w) && !isnothing(b)
        model.latent_proj.layers[1].scale .= w
        model.latent_proj.layers[1].bias .= b
    end
    w = _get_w(weights, "latent_decoder_mean_n_log_scale.1.weight")
    if !isnothing(w)
        model.latent_proj.layers[2].weight .= w
    end

    return model
end
