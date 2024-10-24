module HPCGDebugTests

using PartitionedArrays

include(joinpath("..", "hpcg_benchmark_tests.jl"))

with_debug(hpcg_benchmark_tests)

end # module
