# GPU optimization module for LaProteina
# Routes through Onion dispatch hooks → OnionTile cuTile kernels when available.
#
# Modes (controlled by environment variables, evaluated at precompile time):
#   Default:                OnionTile cuTile kernels + CuArray layer overrides
#   LAPROTEINA_NOCUTILE=1:  Onion dispatch (ONIONop GPU fallback) + CuArray layer overrides
#   LAPROTEINA_NO_OVERRIDES=1: Only TF32 + utils, NO CuArray layer overrides

import ChainRulesCore as CRC
using ChainRulesCore: NoTangent, unthunk

# Onion provides the dispatch hooks (layernorm_first_forward, flash_attention_*_forward)
# OnionTile overrides them with cuTile CuArray methods + rrules
using Onion: layernorm_first_forward, flash_attention_forward, flash_attention_bias_forward

# ONIONop provides within_gradient (AD-context detection)
using ONIONop: within_gradient

const _NO_OVERRIDES = get(ENV, "LAPROTEINA_NO_OVERRIDES", "") != ""
const _FORCE_NOCUTILE = get(ENV, "LAPROTEINA_NOCUTILE", "") != ""

# Check if OnionTile is available (provides cuTile kernel overrides for Onion dispatch)
const _HAS_ONIONTILE = if _NO_OVERRIDES || _FORCE_NOCUTILE
    false
else
    try
        @eval using OnionTile
        true
    catch e
        @warn "OnionTile not available, using Onion dispatch fallbacks" exception=(e, catch_backtrace())
        false
    end
end

if _NO_OVERRIDES
    @info "GPU layer overrides DISABLED via LAPROTEINA_NO_OVERRIDES"
    include("utils_nocutile.jl")
    include("checkpointing.jl")
    include("stubs.jl")
elseif _HAS_ONIONTILE
    # Full optimization path: Onion dispatch hooks → OnionTile cuTile kernels
    include("utils_nocutile.jl")
    include("layers.jl")
    include("checkpointing.jl")
else
    @info "OnionTile disabled, using Onion dispatch fallback (ONIONop kernels)"
    include("utils_nocutile.jl")
    include("layers_nocutile.jl")
    include("checkpointing.jl")
end

# Compatibility alias: scripts reference _HAS_CUTILE, now equivalent to _HAS_ONIONTILE
const _HAS_CUTILE = _HAS_ONIONTILE

export enable_tf32!, within_gradient, safe_checkpointed, forward_from_raw_features_gpu
