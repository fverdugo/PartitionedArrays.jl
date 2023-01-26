module MPIDataPRangeTests

using PartitionedArrays

include(joinpath("..","..","p_range_tests.jl"))

with_mpi_data(p_range_tests)

end # module

