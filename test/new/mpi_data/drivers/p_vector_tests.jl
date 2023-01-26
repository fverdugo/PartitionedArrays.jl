module MPIDataPVectorTests

using PartitionedArrays

include(joinpath("..","..","p_vector_tests.jl"))

with_mpi_data(p_vector_tests)

end # module

