module PartitionedSolversTests

using PartitionedArrays
using PartitionedSolvers
using Test

@testset "smoothers" begin include("smoothers_tests.jl") end

end # module
