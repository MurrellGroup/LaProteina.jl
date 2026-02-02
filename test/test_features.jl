# Tests for features module

@testset "Features" begin
    @testset "Time Embedding" begin
        # Test basic sinusoidal embedding
        t = Float32[0.0, 0.5, 1.0]  # [B=3]
        emb = get_time_embedding(t, 64)

        @test size(emb) == (64, 3)  # [dim, B]
        @test eltype(emb) == Float32

        # t=0 and t=1 should give different embeddings
        @test !isapprox(emb[:, 1], emb[:, 3])

        # Test higher dimension
        emb2 = get_time_embedding(t, 256)
        @test size(emb2) == (256, 3)
    end

    @testset "Time Sampling" begin
        n = 1000

        # Test uniform sampling
        t_uniform = sample_t_uniform(n)
        @test length(t_uniform) == n
        @test all(0 .<= t_uniform .<= 1)
        @test eltype(t_uniform) == Float32

        # Test mixture sampling
        t_mix = sample_t_mix_unif_beta(0.5f0, 2.0f0, 5.0f0, n)
        @test length(t_mix) == n
        @test all(0 .<= t_mix .<= 1)

        # Statistics: mixture should be biased toward t=1
        @test mean(t_mix) > 0.5  # Beta(2,5) is right-skewed when flipped
    end

    @testset "GT Schedule" begin
        # Test tan mode
        gt_tan = gt_schedule(0.5f0, :tan, 0.1f0)
        @test gt_tan > 0

        # Test const mode
        gt_const = gt_schedule(0.5f0, :const, 0.2f0)
        @test gt_const ≈ 0.2f0

        # Test linear mode
        gt_linear_0 = gt_schedule(0.0f0, :linear, 0.1f0)
        gt_linear_1 = gt_schedule(1.0f0, :linear, 0.1f0)
        @test gt_linear_0 ≈ 0.1f0
        @test gt_linear_1 ≈ 0.0f0
    end

    @testset "Inference Time Steps" begin
        steps = inference_time_steps(10)
        @test length(steps) == 11  # n_steps + 1
        @test steps[1] ≈ 0.0f0
        @test steps[end] ≈ 1.0f0

        # Should be monotonically increasing
        for i in 2:length(steps)
            @test steps[i] > steps[i-1]
        end

        # Test with different schedules
        steps_linear = inference_time_steps(10; schedule=:linear)
        @test all(diff(steps_linear) .≈ 0.1f0)  # Uniform spacing
    end

    @testset "Pair Features" begin
        # Create simple CA coordinates
        L, B = 5, 2
        ca_coords = randn(Float32, 3, L, B)

        # Test distance binning
        dist_bins = bin_pairwise_distances(ca_coords; num_bins=16, min_dist=0.0f0, max_dist=2.0f0)
        @test size(dist_bins) == (16, L, L, B)  # [num_bins, L, L, B]

        # Each position should have exactly one bin active (one-hot)
        for b in 1:B, i in 1:L, j in 1:L
            @test sum(dist_bins[:, i, j, b]) ≈ 1.0f0
        end

        # Test relative sequence separation
        rel_sep = relative_sequence_separation(L, B; num_bins=32)
        @test size(rel_sep) == (32, L, L, B)

        # Diagonal should be zero separation (encoded in middle bin)
        # Adjacent positions should be ±1
    end
end
