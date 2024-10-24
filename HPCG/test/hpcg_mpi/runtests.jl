module HPCGMPIRunTests

using Test
using PartitionedArrays

@testset "hpcg_mpi" begin
    include("hpcg_benchmark_tests.jl")
end

end #module
