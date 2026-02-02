# Tests for neural network layers

using Flux

@testset "Neural Network Layers" begin
    @testset "ProteINAAdaLN" begin
        dim = 64
        dim_cond = 32
        L, B = 10, 2

        adaln = ProteINAAdaLN(dim, dim_cond)

        x = randn(Float32, dim, L, B)
        cond = randn(Float32, dim_cond, L, B)
        mask = ones(Float32, L, B)

        y = adaln(x, cond, mask)

        @test size(y) == size(x)
        @test eltype(y) == Float32

        # Output should be masked
        mask[5:L, :] .= 0.0
        y_masked = adaln(x, cond, mask)
        @test all(y_masked[:, 5:L, :] .== 0.0f0)
    end

    @testset "AdaptiveOutputScale" begin
        dim = 64
        dim_cond = 32
        L, B = 10, 2

        scale = AdaptiveOutputScale(dim, dim_cond)

        x = randn(Float32, dim, L, B)
        cond = randn(Float32, dim_cond, L, B)
        mask = ones(Float32, L, B)

        y = scale(x, cond, mask)

        @test size(y) == size(x)
        @test eltype(y) == Float32

        # With default initialization (bias=-2), sigmoid gives small values
        # So output should be scaled down from input
        @test maximum(abs.(y)) < maximum(abs.(x))
    end

    @testset "SwiGLU" begin
        dim = 64
        hidden = 128
        L, B = 10, 2

        swiglu = SwiGLU(dim, hidden)

        x = randn(Float32, dim, L, B)
        y = swiglu(x)

        @test size(y) == (dim, L, B)
    end

    @testset "SwiGLUTransition" begin
        dim = 64
        L, B = 10, 2

        transition = SwiGLUTransition(dim; expansion_factor=4)

        x = randn(Float32, dim, L, B)
        y = transition(x)

        @test size(y) == size(x)
    end

    @testset "TransitionADALN" begin
        dim = 64
        dim_cond = 32
        L, B = 10, 2

        transition = TransitionADALN(dim, dim_cond; expansion_factor=4)

        x = randn(Float32, dim, L, B)
        cond = randn(Float32, dim_cond, L, B)
        mask = ones(Float32, L, B)

        y = transition(x, cond, mask)

        @test size(y) == size(x)
    end

    @testset "ConditioningTransition" begin
        dim = 64
        L, B = 10, 2

        cond_trans = ConditioningTransition(dim; expansion_factor=2)

        x = randn(Float32, dim, L, B)
        mask = ones(Float32, L, B)

        y = cond_trans(x, mask)

        @test size(y) == size(x)
    end

    @testset "PairBiasAttention" begin
        dim_token = 64
        dim_pair = 32
        n_heads = 4
        L, B = 10, 2

        attn = PairBiasAttention(dim_token, dim_pair, n_heads)

        x = randn(Float32, dim_token, L, B)
        pair = randn(Float32, dim_pair, L, L, B)
        mask = ones(Float32, L, B)

        y = attn(x, pair, mask)

        @test size(y) == size(x)
    end

    @testset "MultiHeadBiasedAttentionADALN" begin
        dim_token = 64
        dim_pair = 32
        dim_cond = 32
        n_heads = 4
        L, B = 10, 2

        attn = MultiHeadBiasedAttentionADALN(dim_token, dim_pair, n_heads, dim_cond)

        x = randn(Float32, dim_token, L, B)
        pair = randn(Float32, dim_pair, L, L, B)
        cond = randn(Float32, dim_cond, L, B)
        mask = ones(Float32, L, B)

        y = attn(x, pair, cond, mask)

        @test size(y) == size(x)
    end

    @testset "TransformerBlock" begin
        dim_token = 64
        dim_pair = 32
        dim_cond = 32
        n_heads = 4
        L, B = 10, 2

        block = TransformerBlock(
            dim_token=dim_token,
            dim_pair=dim_pair,
            n_heads=n_heads,
            dim_cond=dim_cond
        )

        x = randn(Float32, dim_token, L, B)
        pair = randn(Float32, dim_pair, L, L, B)
        cond = randn(Float32, dim_cond, L, B)
        mask = ones(Float32, L, B)

        y = block(x, pair, cond, mask)

        @test size(y) == size(x)
    end

    @testset "PairUpdate" begin
        dim_token = 64
        dim_pair = 32
        L, B = 10, 2

        pair_update = PairUpdate(dim_token, dim_pair)

        x = randn(Float32, dim_token, L, B)
        pair = randn(Float32, dim_pair, L, L, B)
        mask = ones(Float32, L, B)

        new_pair = pair_update(x, pair, mask)

        @test size(new_pair) == size(pair)
    end
end
