# Stubs for when CuArray layer overrides are disabled (LAPROTEINA_NO_OVERRIDES=1).
# Provides minimal definitions so that code calling the GPU-optimized forward path
# still works, but delegates to the original (unoptimized) implementations.

"""
    _transformer_block_prenormed(block, x, pair_rep, pair_normed, cond, mask)

Stub: ignores pair_normed and calls the regular TransformerBlock forward.
"""
function _transformer_block_prenormed(block::TransformerBlock, x, pair_rep, pair_normed, cond, mask)
    return block(x, pair_rep, cond, mask)
end

"""
    forward_from_raw_features_gpu(model, raw_features)

Stub: delegates to the standard forward_from_raw_features.
"""
function forward_from_raw_features_gpu(model::ScoreNetwork, raw_features::ScoreNetworkRawFeatures)
    return forward_from_raw_features(model, raw_features)
end
