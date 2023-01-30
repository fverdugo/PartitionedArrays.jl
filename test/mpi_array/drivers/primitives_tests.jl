module MPIArrayPrimitivesTests

using PartitionedArrays

include(joinpath("..","..","primitives_tests.jl"))

with_mpi(primitives_tests)

end # module

