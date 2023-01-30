module MPIArrayPVectorTests

using PartitionedArrays

include(joinpath("..","..","p_vector_tests.jl"))

with_mpi(p_vector_tests)

end # module

