# Tests for utils.jl

using LinearAlgebra

@testset "Utils" begin
    @testset "Tensor Conversion" begin
        # Test python_to_julia for 3D arrays [B,L,D] -> [D,L,B]
        x_py = randn(Float32, 2, 10, 64)  # [B=2, L=10, D=64]
        x_jl = python_to_julia(x_py)
        @test size(x_jl) == (64, 10, 2)  # [D=64, L=10, B=2]
        @test x_jl[1, 1, 1] == x_py[1, 1, 1]
        @test x_jl[64, 10, 2] == x_py[2, 10, 64]

        # Test julia_to_python for 3D arrays [D,L,B] -> [B,L,D]
        y_jl = randn(Float32, 64, 10, 2)
        y_py = julia_to_python(y_jl)
        @test size(y_py) == (2, 10, 64)
        @test y_py[1, 1, 1] == y_jl[1, 1, 1]

        # Test round-trip
        @test python_to_julia(julia_to_python(y_jl)) == y_jl
        @test julia_to_python(python_to_julia(x_py)) == x_py
    end

    @testset "Pair Tensor Conversion" begin
        # Test python_to_julia_pair for 4D arrays [B,N,N,D] -> [D,N,N,B]
        x_py = randn(Float32, 2, 10, 10, 32)  # [B=2, N=10, N=10, D=32]
        x_jl = python_to_julia_pair(x_py)
        @test size(x_jl) == (32, 10, 10, 2)  # [D=32, N=10, N=10, B=2]
        @test x_jl[1, 1, 1, 1] == x_py[1, 1, 1, 1]
        @test x_jl[32, 10, 10, 2] == x_py[2, 10, 10, 32]

        # Test julia_to_python_pair
        y_jl = randn(Float32, 32, 10, 10, 2)
        y_py = julia_to_python_pair(y_jl)
        @test size(y_py) == (2, 10, 10, 32)

        # Round-trip
        @test python_to_julia_pair(julia_to_python_pair(y_jl)) == y_jl
    end

    @testset "Center of Mass" begin
        # Simple test: 3 points along x-axis
        coords = zeros(Float32, 3, 3, 1)  # [3, L=3, B=1]
        coords[1, :, 1] = [-1.0, 0.0, 1.0]  # x-coords: -1, 0, 1

        com = center_of_mass(coords)
        @test size(com) == (3, 1, 1)
        @test com[1, 1, 1] ≈ 0.0f0
        @test com[2, 1, 1] ≈ 0.0f0
        @test com[3, 1, 1] ≈ 0.0f0

        # Test with mask
        coords2 = zeros(Float32, 3, 4, 1)
        coords2[1, :, 1] = [0.0, 1.0, 2.0, 100.0]  # Last point is masked out
        mask = Float32[1.0, 1.0, 1.0, 0.0][:, :]  # [4, 1]

        com2 = center_of_mass(coords2; mask=mask)
        @test com2[1, 1, 1] ≈ 1.0f0  # Mean of [0, 1, 2]
    end

    @testset "Zero Center of Mass" begin
        # Random coordinates
        coords = randn(Float32, 3, 10, 2)

        # Zero COM
        centered = zero_center_of_mass(coords)

        # Check COM is zero for each batch
        for b in 1:2
            com_b = mean(centered[:, :, b]; dims=2)
            @test all(abs.(com_b) .< 1e-5)
        end

        # With mask
        mask = ones(Float32, 10, 2)
        mask[8:10, :] .= 0.0

        centered_masked = zero_center_of_mass(coords; mask=mask)
        # COM over unmasked positions should be ~zero
        for b in 1:2
            unmasked_coords = centered_masked[:, 1:7, b]
            com_b = mean(unmasked_coords; dims=2)
            @test all(abs.(com_b) .< 1e-5)
        end
    end

    @testset "Expand Mask" begin
        mask = ones(Float32, 10, 2)  # [L, B]

        # Expand to 3D [1, L, B]
        mask_3d = expand_mask(mask, 3)
        @test size(mask_3d) == (1, 10, 2)

        # Expand to 4D [1, 1, L, B]
        mask_4d = expand_mask(mask, 4)
        @test size(mask_4d) == (1, 1, 10, 2)
    end
end
