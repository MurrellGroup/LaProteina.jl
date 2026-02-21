# GPU optimization module for LaProteina
# Routes through Onion dispatch hooks → OnionTile cuTile kernels at runtime.
#
# Modes (controlled by environment variables, evaluated at precompile time):
#   Default:                    Full CuArray layer overrides (layers.jl)
#   LAPROTEINA_NO_OVERRIDES=1: Only TF32 + utils + stubs (no CuArray overrides)

import ChainRulesCore as CRC
using ChainRulesCore: NoTangent, unthunk

# Onion provides the dispatch hooks (layernorm_first_forward, flash_attention_*_forward)
# OnionTile overrides them with cuTile CuArray methods + rrules when loaded in the run env
using Onion: layernorm_first_forward, flash_attention_forward, flash_attention_bias_forward

# ONIONop provides within_gradient (AD-context detection)
using ONIONop: within_gradient

const _NO_OVERRIDES = get(ENV, "LAPROTEINA_NO_OVERRIDES", "") != ""

if _NO_OVERRIDES
    @info "GPU layer overrides DISABLED via LAPROTEINA_NO_OVERRIDES"
    include("utils_nocutile.jl")
    include("checkpointing.jl")
    include("stubs.jl")
else
    include("utils_nocutile.jl")
    include("layers.jl")
    include("checkpointing.jl")
end

export enable_tf32!, within_gradient, safe_checkpointed, forward_from_raw_features_gpu
