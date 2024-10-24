module HPCGMPITests

using PartitionedArrays

include(joinpath("..", "..", "hpcg_benchmark_tests.jl"))

with_mpi(hpcg_benchmark_tests)

end # module
