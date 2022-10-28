module PartitionedArraysRunTests

using Test

@testset "jagged_array" begin include("jagged_array_tests.jl") end

@testset "sequential_array" begin include("sequential_array/runtests.jl") end

end # module
