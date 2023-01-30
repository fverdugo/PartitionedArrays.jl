module PartitionedArraysRunTests

using Test

@testset "jagged_array" begin include("jagged_array_tests.jl") end
@testset "sparse_utils" begin include("sparse_utils_tests.jl") end
@testset "debug_array" begin include("debug_array/runtests.jl") end
@testset "mpi_array" begin include("mpi_array/runtests.jl") end

end # module
