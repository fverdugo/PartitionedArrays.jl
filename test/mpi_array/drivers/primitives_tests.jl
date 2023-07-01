module MPIArrayPrimitivesTests

using PartitionedArrays
using MPI 

include(joinpath("..","..","primitives_tests.jl"))

with_mpi(primitives_tests)

end # module

