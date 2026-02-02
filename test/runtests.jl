# JuProteina Test Suite

using Test
using JuProteina

@testset "JuProteina Tests" begin
    include("test_constants.jl")
    include("test_utils.jl")
    include("test_features.jl")
    include("test_layers.jl")

    # Parity tests require PyCall and Python environment
    if get(ENV, "RUN_PARITY_TESTS", "false") == "true"
        include("test_parity.jl")
    end
end
