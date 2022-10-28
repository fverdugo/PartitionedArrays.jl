module SequentialArrayRunTests

using Test
using PartitionedArrays

@testset "sequential_array" begin include("sequential_array_tests.jl") end

@testset "primitives" begin include("primitives_tests.jl")  end

end #module
