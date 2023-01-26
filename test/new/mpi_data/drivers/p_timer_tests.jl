module MPIDataPrimitivesTests

using PartitionedArrays

include(joinpath("..","..","p_timer_tests.jl"))

with_mpi_data(p_timer_tests)

end # module

