# Safe gradient checkpointing for use with in-place CuArray layer overrides.
#
# Problem: Zygote.checkpointed runs the forward inside a CRC rrule (not traced),
# so within_gradient() returns false. This causes CuArray layers to use the
# inference path with in-place mutations, corrupting saved inputs.
#
# Solution: safe_checkpointed copies inputs before the forward call, so backward
# recomputation uses the original (unmutated) values.

import Zygote

"""
    safe_checkpointed(f, args...)

Like `Zygote.checkpointed` but copies array inputs before the forward call.
Prevents in-place mutations during inference-path forward from corrupting
saved inputs for backward recomputation.
"""
function safe_checkpointed(f, args...; kwargs...)
    return f(args...; kwargs...)
end

function CRC.rrule(::typeof(safe_checkpointed), f, args...; kwargs...)
    saved_args = map(copy, args)
    y = f(args...; kwargs...)
    function safe_checkpoint_pullback(dy)
        dy_val = if dy isa CRC.Tangent
            map(unthunk, Tuple(dy))
        else
            unthunk(dy)
        end
        _, back = Zygote.pullback(f, saved_args...; kwargs...)
        grads = back(dy_val)
        return (NoTangent(), NoTangent(), grads...)
    end
    return y, safe_checkpoint_pullback
end
