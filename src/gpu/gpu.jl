# GPU optimization module for LaProteina
# Provides: TF32, within_gradient, buffer pooling, CuArray method overrides.
# When cuTile is available: adds fused LayerNorm and Flash Attention kernels.
#
# Modes (controlled by environment variables, evaluated at precompile time):
#   Default:                cuTile kernels + CuArray layer overrides
#   LAPROTEINA_NOCUTILE=1:  NNlib attention + CuArray layer overrides (no cuTile)
#   LAPROTEINA_NO_OVERRIDES=1: Only TF32 + utils, NO CuArray layer overrides

import ChainRulesCore as CRC
using ChainRulesCore: NoTangent, unthunk

const _NO_OVERRIDES = get(ENV, "LAPROTEINA_NO_OVERRIDES", "") != ""
const _FORCE_NOCUTILE = get(ENV, "LAPROTEINA_NOCUTILE", "") != ""

# Check if cuTile is available (skip if forced nocutile or no overrides)
const _HAS_CUTILE = if _NO_OVERRIDES || _FORCE_NOCUTILE
    false
else
    try
        @eval import cuTile as ct
        true
    catch e
        @warn "cuTile not available" exception=(e, catch_backtrace())
        false
    end
end

if _NO_OVERRIDES
    @info "GPU layer overrides DISABLED via LAPROTEINA_NO_OVERRIDES"
    include("utils_nocutile.jl")
    include("checkpointing.jl")
    include("stubs.jl")
elseif _HAS_CUTILE
    # Full optimization path with cuTile kernels
    include("utils.jl")
    include("kernels/layernorm.jl")
    include("kernels/fmha.jl")
    include("kernels/fmha_bwd.jl")
    include("dispatch.jl")
    include("rrules.jl")
    include("layers.jl")
    include("checkpointing.jl")
else
    @info "cuTile disabled, using NNlib attention path"
    include("utils_nocutile.jl")
    include("layers_nocutile.jl")
    include("checkpointing.jl")
end

export enable_tf32!, within_gradient, safe_checkpointed, forward_from_raw_features_gpu
