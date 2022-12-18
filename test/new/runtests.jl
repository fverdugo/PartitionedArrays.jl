module PartitionedArraysRunTests

using Test

@testset "jagged_array" begin include("jagged_array_tests.jl") end

@testset "partition" begin include("partition_tests.jl") end

@testset "sequential_data" begin include("sequential_data/runtests.jl") end

@testset "mpi_data" begin include("mpi_data/runtests.jl") end

end # module
