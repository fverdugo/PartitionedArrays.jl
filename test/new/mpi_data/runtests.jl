module MPIDataRunTests

using Test
using PartitionedArrays

@testset "mpi_data" begin include("mpi_data_tests.jl") end

#@testset "primitives" begin include("primitives_tests.jl")  end

end #module
