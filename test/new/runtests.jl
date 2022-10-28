module PartitionedArraysTests

using Test

@testset "jagged_array" begin include("jagged_array_tests.jl") end

@testset "sequential" begin include("sequential/runtests.jl") end

end # module
