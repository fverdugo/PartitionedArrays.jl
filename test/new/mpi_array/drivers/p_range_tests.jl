module MPIArrayPRangeTests

using PartitionedArrays

include(joinpath("..","..","p_range_tests.jl"))

with_mpi(p_range_tests)

end # module

