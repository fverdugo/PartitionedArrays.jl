module MPIDataPrimitivesTests

using PartitionedArrays

include(joinpath("..","..","p_timer_tests.jl"))

with_mpi(p_timer_tests)

end # module

