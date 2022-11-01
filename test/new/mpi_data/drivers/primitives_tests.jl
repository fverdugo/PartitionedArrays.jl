module MPIDataPrimitivesTests

using PartitionedArrays

include(joinpath("..","..","primitives_tests.jl"))

with_mpi_data(primitives_tests)

end # module

