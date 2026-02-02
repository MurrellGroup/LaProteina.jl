# Tests for constants.jl

@testset "Constants" begin
    @testset "Atom Types" begin
        @test length(ATOM_TYPES) == 37
        @test ATOM_TYPES[1] == "N"
        @test ATOM_TYPES[2] == "CA"
        @test ATOM_TYPES[3] == "C"
        @test ATOM_TYPES[4] == "CB"
        @test ATOM_TYPES[5] == "O"

        @test CA_INDEX == 2
        @test CB_INDEX == 4
    end

    @testset "Amino Acids" begin
        @test length(RESTYPES) == 20
        @test RESTYPES[1] == 'A'  # Alanine
        @test RESTYPES[20] == 'V'  # Valine

        # Test index conversions
        @test aa_to_index('A') == 1
        @test aa_to_index('V') == 20
        @test aa_to_index('X') == 21  # Unknown

        @test index_to_aa(1) == 'A'
        @test index_to_aa(20) == 'V'
        @test index_to_aa(21) == 'X'

        # Test 3-letter conversions
        @test RESTYPE_3TO1["ALA"] == 'A'
        @test RESTYPE_3TO1["VAL"] == 'V'
        @test RESTYPE_1TO3['A'] == "ALA"
        @test RESTYPE_1TO3['V'] == "VAL"
    end

    @testset "Atom Order" begin
        @test ATOM_ORDER["N"] == 1
        @test ATOM_ORDER["CA"] == 2
        @test ATOM_ORDER["C"] == 3
        @test ATOM_ORDER["CB"] == 4
        @test ATOM_ORDER["O"] == 5
    end

    @testset "Atom Masks" begin
        # RESTYPE_ATOM37_MASK should be [21, 37]
        @test size(RESTYPE_ATOM37_MASK) == (21, 37)

        # All residues should have backbone atoms (N, CA, C, O)
        for i in 1:20
            @test RESTYPE_ATOM37_MASK[i, 1] == true  # N
            @test RESTYPE_ATOM37_MASK[i, 2] == true  # CA
            @test RESTYPE_ATOM37_MASK[i, 3] == true  # C
            @test RESTYPE_ATOM37_MASK[i, 5] == true  # O
        end

        # Glycine (G, index 7) should not have CB
        @test RESTYPE_ATOM37_MASK[7, 4] == false  # CB

        # Alanine (A, index 1) should have CB but no other sidechain
        @test RESTYPE_ATOM37_MASK[1, 4] == true  # CB
    end
end
