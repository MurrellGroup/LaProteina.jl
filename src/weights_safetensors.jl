# SafeTensors weight loading utilities for pretrained models
# Parallel to weights.jl (NPZ-based), adapted for PyTorch state_dict key names

using Flux

# Conditionally load SafeTensors
# SafeTensors.jl provides: load_safetensors(path) → Dict{String, Array}
import SafeTensors

# ============================================================================
# Tracking infrastructure — catches wrong prefix, missing/extra keys
# ============================================================================

"""Track which keys were consumed (loaded) and which were looked up but missing."""
mutable struct _STLoadTracker
    consumed::Set{String}   # Keys found and loaded
    missed::Set{String}     # Keys looked up but not found
end
_STLoadTracker() = _STLoadTracker(Set{String}(), Set{String}())

const _st_tracker = Ref(_STLoadTracker())

"""
    _st_get(weights::Dict, key::String)

Retrieve a weight tensor from SafeTensors dict, converting to Float32.
Returns nothing if key doesn't exist. Tracks all lookups for validation.
"""
function _st_get(weights::Dict, key::String)
    if haskey(weights, key)
        push!(_st_tracker[].consumed, key)
        return Float32.(weights[key])
    else
        push!(_st_tracker[].missed, key)
        return nothing
    end
end
_st_get(weights::Dict, prefix::String, key::String) = _st_get(weights, prefix * key)

"""
    _load_safetensors(path::String) -> Dict{String, Array}

Load a .safetensors file and return all tensors as a Dict.
"""
function _load_safetensors(path::String)
    return SafeTensors.load_safetensors(path)
end

"""
    _validate_st_loading(weights::Dict, prefix::String, model_type::String)

Check for mismatches between file keys and loaded keys. Errors on any discrepancy.
"""
function _validate_st_loading(weights::Dict, prefix::String, model_type::String)
    tracker = _st_tracker[]

    # Keys in file matching our prefix
    file_prefix_keys = Set(k for k in keys(weights) if startswith(k, prefix))

    # Keys in file with our prefix that we never loaded
    unused = sort(collect(setdiff(file_prefix_keys, tracker.consumed)))

    # Keys we looked for (with our prefix) but didn't find
    missing_keys = sort(collect(tracker.missed))

    n_loaded = length(tracker.consumed)
    errors = String[]

    if n_loaded == 0 && !isempty(file_prefix_keys)
        top_prefixes = sort(unique(first(split(k, ".")) for k in keys(weights)))
        push!(errors, "ZERO weights loaded! No keys matched prefix \"$(prefix)\". " *
              "File has $(length(file_prefix_keys)) keys with this prefix but none were consumed. " *
              "Available top-level prefixes: $(join(top_prefixes, ", "))")
    elseif n_loaded == 0
        top_prefixes = sort(unique(first(split(k, ".")) for k in keys(weights)))
        push!(errors, "ZERO weights loaded! No keys in file match prefix \"$(prefix)\". " *
              "Available top-level prefixes: $(join(top_prefixes, ", "))")
    end

    if !isempty(unused)
        shown = unused[1:min(20, length(unused))]
        push!(errors, "$(length(unused)) weight(s) in file (prefix \"$prefix\") were NOT loaded into $model_type:" *
              "\n    " * join(shown, "\n    ") *
              (length(unused) > 20 ? "\n    ... and $(length(unused) - 20) more" : ""))
    end

    if !isempty(missing_keys)
        shown = missing_keys[1:min(20, length(missing_keys))]
        push!(errors, "$(length(missing_keys)) weight(s) expected by $model_type were NOT found in file:" *
              "\n    " * join(shown, "\n    ") *
              (length(missing_keys) > 20 ? "\n    ... and $(length(missing_keys) - 20) more" : ""))
    end

    if !isempty(errors)
        error("Weight loading mismatch for $model_type ($n_loaded weights loaded):\n\n" *
              join(errors, "\n\n"))
    end

    @info "$model_type: loaded $n_loaded weights from $(length(file_prefix_keys)) file keys (prefix=\"$prefix\")"
end

# ============================================================================
# Score Network weight loading (SafeTensors)
# ============================================================================

"""
    load_score_network_weights_st!(model::ScoreNetwork, weights_path::String;
                                    prefix::String="nn.", strict::Bool=true)

Load pretrained weights into ScoreNetwork from SafeTensors file.
PyTorch Lightning state_dict keys have an "nn." prefix by default.

If `strict=true` (default), errors on any mismatch between file keys and model parameters.
"""
function load_score_network_weights_st!(model::ScoreNetwork, weights_path::String;
                                         prefix::String="nn.", strict::Bool=true)
    weights = _load_safetensors(weights_path)
    _st_tracker[] = _STLoadTracker()  # Reset tracker

    # init_repr_factory: sequence feature projection
    w = _st_get(weights, prefix, "init_repr_factory.linear_out.weight")
    if !isnothing(w)
        model.init_repr_factory.projection.weight .= w
    end

    # cond_factory: conditioning projection
    w = _st_get(weights, prefix, "cond_factory.linear_out.weight")
    if !isnothing(w)
        model.cond_factory.projection.weight .= w
    end

    # motif_uidx_factory (LD6/LD7 unindexed motif projection)
    if !isnothing(model.motif_uidx_proj)
        w = _st_get(weights, prefix, "motif_uidx_factory.linear_out.weight")
        if !isnothing(w)
            model.motif_uidx_proj.weight .= w
        end
    end

    # pair_repr_builder
    _load_pair_repr_builder_st!(model.pair_rep_builder, weights, prefix * "pair_repr_builder.")

    # transition_c_1
    _load_conditioning_transition_st!(model.transition_c_1, weights, prefix * "transition_c_1.")

    # transition_c_2
    _load_conditioning_transition_st!(model.transition_c_2, weights, prefix * "transition_c_2.")

    # transformer_layers
    for (i, layer) in enumerate(model.transformer_layers)
        lprefix = prefix * "transformer_layers.$(i-1)."
        _load_transformer_block_st!(layer, weights, lprefix)
    end

    # pair_update_layers (including triangle multiplicative updates)
    if model.update_pair_repr
        for (i, pu) in enumerate(model.pair_update_layers)
            if !isnothing(pu)
                lprefix = prefix * "pair_update_layers.$(i-1)."
                _load_pair_update_st!(pu, weights, lprefix)
            end
        end
    end

    # Output projections
    # local_latents_proj: LayerNorm + Dense
    w = _st_get(weights, prefix, "local_latents_linear.0.weight")
    b = _st_get(weights, prefix, "local_latents_linear.0.bias")
    if !isnothing(w) && !isnothing(b)
        model.local_latents_proj.layers[1].scale .= w
        model.local_latents_proj.layers[1].bias .= b
    end
    w = _st_get(weights, prefix, "local_latents_linear.1.weight")
    if !isnothing(w)
        model.local_latents_proj.layers[2].weight .= w
    end

    # ca_proj: LayerNorm + Dense
    w = _st_get(weights, prefix, "ca_linear.0.weight")
    b = _st_get(weights, prefix, "ca_linear.0.bias")
    if !isnothing(w) && !isnothing(b)
        model.ca_proj.layers[1].scale .= w
        model.ca_proj.layers[1].bias .= b
    end
    w = _st_get(weights, prefix, "ca_linear.1.weight")
    if !isnothing(w)
        model.ca_proj.layers[2].weight .= w
    end

    if strict
        _validate_st_loading(weights, prefix, "ScoreNetwork")
    end
    return model
end

# ============================================================================
# Component-level weight loading helpers
# ============================================================================

function _load_pair_repr_builder_st!(builder::PairReprBuilder, weights::Dict, prefix::String)
    # init_repr_factory
    w = _st_get(weights, prefix, "init_repr_factory.linear_out.weight")
    if !isnothing(w)
        builder.init_repr_factory.projection.weight .= w
    end

    # LayerNorm if present
    if builder.init_repr_factory.use_ln && !isnothing(builder.init_repr_factory.ln)
        w = _st_get(weights, prefix, "init_repr_factory.ln_out.weight")
        b = _st_get(weights, prefix, "init_repr_factory.ln_out.bias")
        if !isnothing(w) && !isnothing(b)
            builder.init_repr_factory.ln.scale .= w
            builder.init_repr_factory.ln.bias .= b
        end
    end

    # cond_factory
    if !isnothing(builder.cond_factory)
        w = _st_get(weights, prefix, "cond_factory.linear_out.weight")
        if !isnothing(w)
            builder.cond_factory.projection.weight .= w
        end
        if builder.cond_factory.use_ln && !isnothing(builder.cond_factory.ln)
            w = _st_get(weights, prefix, "cond_factory.ln_out.weight")
            b = _st_get(weights, prefix, "cond_factory.ln_out.bias")
            if !isnothing(w) && !isnothing(b)
                builder.cond_factory.ln.scale .= w
                builder.cond_factory.ln.bias .= b
            end
        end
    end

    # adaln
    if !isnothing(builder.adaln)
        _load_adaln_identical_st!(builder.adaln, weights, prefix * "adaln.")
    end
end

function _load_adaln_identical_st!(adaln::AdaptiveLayerNormIdentical, weights::Dict, prefix::String)
    # norm_cond
    if !isnothing(adaln.norm_cond)
        w = _st_get(weights, prefix, "norm_cond.weight")
        b = _st_get(weights, prefix, "norm_cond.bias")
        if !isnothing(w) && !isnothing(b)
            adaln.norm_cond.scale .= w
            adaln.norm_cond.bias .= b
        end
    end

    # to_gamma: Dense + sigmoid (in Chain)
    w = _st_get(weights, prefix, "to_gamma.0.weight")
    b = _st_get(weights, prefix, "to_gamma.0.bias")
    if !isnothing(w)
        adaln.to_gamma.layers[1].weight .= w
        if !isnothing(b)
            adaln.to_gamma.layers[1].bias .= b
        end
    end

    # to_beta
    w = _st_get(weights, prefix, "to_beta.weight")
    if !isnothing(w)
        adaln.to_beta.weight .= w
    end
end

function _load_conditioning_transition_st!(trans::ConditioningTransition, weights::Dict, prefix::String)
    # swish_linear.0 -> linear_in
    w = _st_get(weights, prefix, "swish_linear.0.weight")
    if !isnothing(w)
        trans.transition.linear_in.weight .= w
    end

    # linear_out
    w = _st_get(weights, prefix, "linear_out.weight")
    if !isnothing(w)
        trans.transition.linear_out.weight .= w
    end
end

function _load_transformer_block_st!(block::TransformerBlock, weights::Dict, prefix::String)
    # mhba (MultiHeadBiasedAttentionADALN)
    mhba_prefix = prefix * "mhba."

    # adaln
    _load_adaln_st!(block.mha.adaln, weights, mhba_prefix * "adaln.")

    # mha (PairBiasAttention)
    mha_prefix = mhba_prefix * "mha."
    _load_pair_bias_attention_st!(block.mha.mha, weights, mha_prefix)

    # scale_output
    _load_adaptive_output_scale_st!(block.mha.scale_output, weights, mhba_prefix * "scale_output.")

    # transition (TransitionADALN)
    trans_prefix = prefix * "transition."

    # adaln
    _load_adaln_st!(block.transition.adaln, weights, trans_prefix * "adaln.")

    # transition (SwiGLUTransition)
    swiglu_prefix = trans_prefix * "transition."
    w = _st_get(weights, swiglu_prefix, "swish_linear.0.weight")
    if !isnothing(w)
        block.transition.transition.linear_in.weight .= w
    end
    w = _st_get(weights, swiglu_prefix, "linear_out.weight")
    if !isnothing(w)
        block.transition.transition.linear_out.weight .= w
    end

    # scale_output
    _load_adaptive_output_scale_st!(block.transition.scale_output, weights, trans_prefix * "scale_output.")
end

function _load_adaln_st!(adaln::ProteINAAdaLN, weights::Dict, prefix::String)
    # norm_cond
    w = _st_get(weights, prefix, "norm_cond.weight")
    b = _st_get(weights, prefix, "norm_cond.bias")
    if !isnothing(w) && !isnothing(b)
        adaln.norm_cond.scale .= w
        adaln.norm_cond.bias .= b
    end

    # to_gamma
    w = _st_get(weights, prefix, "to_gamma.0.weight")
    b = _st_get(weights, prefix, "to_gamma.0.bias")
    if !isnothing(w)
        adaln.to_gamma.layers[1].weight .= w
        if !isnothing(b)
            adaln.to_gamma.layers[1].bias .= b
        end
    end

    # to_beta
    w = _st_get(weights, prefix, "to_beta.weight")
    if !isnothing(w)
        adaln.to_beta.weight .= w
    end
end

function _load_pair_bias_attention_st!(mha::PairBiasAttention, weights::Dict, prefix::String)
    # node_norm
    w = _st_get(weights, prefix, "node_norm.weight")
    b = _st_get(weights, prefix, "node_norm.bias")
    if !isnothing(w) && !isnothing(b)
        mha.node_norm.scale .= w
        mha.node_norm.bias .= b
    end

    # pair_norm
    w = _st_get(weights, prefix, "pair_norm.weight")
    b = _st_get(weights, prefix, "pair_norm.bias")
    if !isnothing(w) && !isnothing(b)
        mha.pair_norm.scale .= w
        mha.pair_norm.bias .= b
    end

    # q_layer_norm, k_layer_norm (if present)
    if !isnothing(mha.q_norm)
        w = _st_get(weights, prefix, "q_layer_norm.weight")
        b = _st_get(weights, prefix, "q_layer_norm.bias")
        if !isnothing(w) && !isnothing(b)
            mha.q_norm.scale .= w
            mha.q_norm.bias .= b
        end
    end
    if !isnothing(mha.k_norm)
        w = _st_get(weights, prefix, "k_layer_norm.weight")
        b = _st_get(weights, prefix, "k_layer_norm.bias")
        if !isnothing(w) && !isnothing(b)
            mha.k_norm.scale .= w
            mha.k_norm.bias .= b
        end
    end

    # to_qkv
    w = _st_get(weights, prefix, "to_qkv.weight")
    b = _st_get(weights, prefix, "to_qkv.bias")
    if !isnothing(w)
        mha.to_qkv.weight .= w
        if !isnothing(b)
            mha.to_qkv.bias .= b
        end
    end

    # to_g
    w = _st_get(weights, prefix, "to_g.weight")
    b = _st_get(weights, prefix, "to_g.bias")
    if !isnothing(w)
        mha.to_g.weight .= w
        if !isnothing(b)
            mha.to_g.bias .= b
        end
    end

    # to_out (called to_out_node in Python)
    w = _st_get(weights, prefix, "to_out_node.weight")
    b = _st_get(weights, prefix, "to_out_node.bias")
    if !isnothing(w)
        mha.to_out.weight .= w
        if !isnothing(b)
            mha.to_out.bias .= b
        end
    end

    # to_bias
    w = _st_get(weights, prefix, "to_bias.weight")
    if !isnothing(w)
        mha.to_bias.weight .= w
    end
end

function _load_adaptive_output_scale_st!(aos::AdaptiveOutputScale, weights::Dict, prefix::String)
    w = _st_get(weights, prefix, "to_adaln_zero_gamma.0.weight")
    b = _st_get(weights, prefix, "to_adaln_zero_gamma.0.bias")
    if !isnothing(w)
        aos.linear.weight .= w
        if !isnothing(b)
            aos.linear.bias .= b
        end
    end
end

# ============================================================================
# PairUpdate weight loading (including triangle multiplicative updates)
# ============================================================================

function _load_pair_update_st!(pu::PairUpdate, weights::Dict, prefix::String)
    # Sequence LayerNorm (layer_norm_in in Python)
    if !isnothing(pu.seq_ln)
        w = _st_get(weights, prefix, "layer_norm_in.weight")
        b = _st_get(weights, prefix, "layer_norm_in.bias")
        if !isnothing(w) && !isnothing(b)
            pu.seq_ln.scale .= w
            pu.seq_ln.bias .= b
        end
    end

    # linear_x (projects tokens to 2*pair_dim, called outer_proj in current code)
    w = _st_get(weights, prefix, "linear_x.weight")
    if !isnothing(w)
        pu.outer_proj.weight .= w
    end

    # Triangle multiplicative out
    if !isnothing(pu.tri_out)
        _load_triangle_multiplication_st!(pu.tri_out, weights, prefix * "tri_mult_out.")
    end

    # Triangle multiplicative in
    if !isnothing(pu.tri_in)
        _load_triangle_multiplication_st!(pu.tri_in, weights, prefix * "tri_mult_in.")
    end

    # Pair transition (always present)
    _load_pair_transition_st!(pu.pair_transition, weights, prefix * "transition_out.")
end

function _load_triangle_multiplication_st!(tri::TriangleMultiplication, weights::Dict, prefix::String)
    # layer_norm_in
    w = _st_get(weights, prefix, "layer_norm_in.weight")
    b = _st_get(weights, prefix, "layer_norm_in.bias")
    if !isnothing(w) && !isnothing(b)
        tri.layer_norm_in.scale .= w
        tri.layer_norm_in.bias .= b
    end

    # linear_a_p
    w = _st_get(weights, prefix, "linear_a_p.weight")
    b = _st_get(weights, prefix, "linear_a_p.bias")
    if !isnothing(w)
        tri.linear_a_p.weight .= w
        if !isnothing(b)
            tri.linear_a_p.bias .= b
        end
    end

    # linear_a_g
    w = _st_get(weights, prefix, "linear_a_g.weight")
    b = _st_get(weights, prefix, "linear_a_g.bias")
    if !isnothing(w)
        tri.linear_a_g.weight .= w
        if !isnothing(b)
            tri.linear_a_g.bias .= b
        end
    end

    # linear_b_p
    w = _st_get(weights, prefix, "linear_b_p.weight")
    b = _st_get(weights, prefix, "linear_b_p.bias")
    if !isnothing(w)
        tri.linear_b_p.weight .= w
        if !isnothing(b)
            tri.linear_b_p.bias .= b
        end
    end

    # linear_b_g
    w = _st_get(weights, prefix, "linear_b_g.weight")
    b = _st_get(weights, prefix, "linear_b_g.bias")
    if !isnothing(w)
        tri.linear_b_g.weight .= w
        if !isnothing(b)
            tri.linear_b_g.bias .= b
        end
    end

    # linear_g
    w = _st_get(weights, prefix, "linear_g.weight")
    b = _st_get(weights, prefix, "linear_g.bias")
    if !isnothing(w)
        tri.linear_g.weight .= w
        if !isnothing(b)
            tri.linear_g.bias .= b
        end
    end

    # linear_z
    w = _st_get(weights, prefix, "linear_z.weight")
    b = _st_get(weights, prefix, "linear_z.bias")
    if !isnothing(w)
        tri.linear_z.weight .= w
        if !isnothing(b)
            tri.linear_z.bias .= b
        end
    end

    # layer_norm_out
    w = _st_get(weights, prefix, "layer_norm_out.weight")
    b = _st_get(weights, prefix, "layer_norm_out.bias")
    if !isnothing(w) && !isnothing(b)
        tri.layer_norm_out.scale .= w
        tri.layer_norm_out.bias .= b
    end
end

function _load_pair_transition_st!(pt::PairTransition, weights::Dict, prefix::String)
    # layer_norm
    w = _st_get(weights, prefix, "layer_norm.weight")
    b = _st_get(weights, prefix, "layer_norm.bias")
    if !isnothing(w) && !isnothing(b)
        pt.layer_norm.scale .= w
        pt.layer_norm.bias .= b
    end

    # linear_1
    w = _st_get(weights, prefix, "linear_1.weight")
    b = _st_get(weights, prefix, "linear_1.bias")
    if !isnothing(w)
        pt.linear_1.weight .= w
        if !isnothing(b)
            pt.linear_1.bias .= b
        end
    end

    # linear_2
    w = _st_get(weights, prefix, "linear_2.weight")
    b = _st_get(weights, prefix, "linear_2.bias")
    if !isnothing(w)
        pt.linear_2.weight .= w
        if !isnothing(b)
            pt.linear_2.bias .= b
        end
    end
end

# ============================================================================
# Encoder weight loading (SafeTensors)
# ============================================================================

"""
    load_encoder_weights_st!(model::EncoderTransformer, weights_path::String;
                              prefix::String="encoder.", strict::Bool=true)

Load pretrained weights into EncoderTransformer from SafeTensors file.

If `strict=true` (default), errors on any mismatch between file keys and model parameters.
"""
function load_encoder_weights_st!(model::EncoderTransformer, weights_path::String;
                                    prefix::String="encoder.", strict::Bool=true)
    weights = _load_safetensors(weights_path)
    _st_tracker[] = _STLoadTracker()  # Reset tracker

    # init_repr_factory
    w = _st_get(weights, prefix, "init_repr_factory.linear_out.weight")
    if !isnothing(w)
        model.init_repr_factory.projection.weight .= w
    end

    # pair_rep_factory
    w = _st_get(weights, prefix, "pair_rep_factory.linear_out.weight")
    if !isnothing(w)
        model.pair_rep_factory.projection.weight .= w
    end

    # transition_c_1, transition_c_2
    _load_conditioning_transition_st!(model.transition_c_1, weights, prefix * "transition_c_1.")
    _load_conditioning_transition_st!(model.transition_c_2, weights, prefix * "transition_c_2.")

    # transformer_layers
    for (i, layer) in enumerate(model.transformer_layers)
        lprefix = prefix * "transformer_layers.$(i-1)."
        _load_transformer_block_st!(layer, weights, lprefix)
    end

    # latent_proj: LayerNorm + Dense
    w = _st_get(weights, prefix, "latent_decoder_mean_n_log_scale.0.weight")
    b = _st_get(weights, prefix, "latent_decoder_mean_n_log_scale.0.bias")
    if !isnothing(w) && !isnothing(b)
        model.latent_proj.layers[1].scale .= w
        model.latent_proj.layers[1].bias .= b
    end
    w = _st_get(weights, prefix, "latent_decoder_mean_n_log_scale.1.weight")
    if !isnothing(w)
        model.latent_proj.layers[2].weight .= w
    end

    if strict
        _validate_st_loading(weights, prefix, "EncoderTransformer")
    end
    return model
end

# ============================================================================
# Decoder weight loading (SafeTensors)
# ============================================================================

"""
    load_decoder_weights_st!(model::DecoderTransformer, weights_path::String;
                              prefix::String="decoder.", strict::Bool=true)

Load pretrained weights into DecoderTransformer from SafeTensors file.

If `strict=true` (default), errors on any mismatch between file keys and model parameters.
"""
function load_decoder_weights_st!(model::DecoderTransformer, weights_path::String;
                                    prefix::String="decoder.", strict::Bool=true)
    weights = _load_safetensors(weights_path)
    _st_tracker[] = _STLoadTracker()  # Reset tracker

    # init_repr_factory
    w = _st_get(weights, prefix, "init_repr_factory.linear_out.weight")
    if !isnothing(w)
        model.init_repr_factory.projection.weight .= w
    end

    # pair_rep_factory
    w = _st_get(weights, prefix, "pair_rep_factory.linear_out.weight")
    if !isnothing(w)
        model.pair_rep_factory.projection.weight .= w
    end

    # transition_c_1, transition_c_2
    _load_conditioning_transition_st!(model.transition_c_1, weights, prefix * "transition_c_1.")
    _load_conditioning_transition_st!(model.transition_c_2, weights, prefix * "transition_c_2.")

    # transformer_layers
    for (i, layer) in enumerate(model.transformer_layers)
        lprefix = prefix * "transformer_layers.$(i-1)."
        _load_transformer_block_st!(layer, weights, lprefix)
    end

    # Output projections
    # logit_proj: LayerNorm + Dense
    w = _st_get(weights, prefix, "logit_linear.0.weight")
    b = _st_get(weights, prefix, "logit_linear.0.bias")
    if !isnothing(w) && !isnothing(b)
        model.logit_proj.layers[1].scale .= w
        model.logit_proj.layers[1].bias .= b
    end
    w = _st_get(weights, prefix, "logit_linear.1.weight")
    if !isnothing(w)
        model.logit_proj.layers[2].weight .= w
    end

    # struct_proj: LayerNorm + Dense
    w = _st_get(weights, prefix, "struct_linear.0.weight")
    b = _st_get(weights, prefix, "struct_linear.0.bias")
    if !isnothing(w) && !isnothing(b)
        model.struct_proj.layers[1].scale .= w
        model.struct_proj.layers[1].bias .= b
    end
    w = _st_get(weights, prefix, "struct_linear.1.weight")
    if !isnothing(w)
        model.struct_proj.layers[2].weight .= w
    end

    if strict
        _validate_st_loading(weights, prefix, "DecoderTransformer")
    end
    return model
end
