module SequentialDataRunTests

using Test
using PartitionedArrays

@testset "sequential_data" begin include("sequential_data_tests.jl") end

@testset "primitives" begin include("primitives_tests.jl")  end

end #module
