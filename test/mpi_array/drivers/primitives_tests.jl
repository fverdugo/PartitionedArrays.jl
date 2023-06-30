module MPIArrayPrimitivesTests

using PartitionedArrays
using MPI 

println(MPI.versioninfo())

include(joinpath("..","..","primitives_tests.jl"))

with_mpi(primitives_tests)

end # module

