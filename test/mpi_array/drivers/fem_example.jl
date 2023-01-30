module MPIArrayFEMExample

using PartitionedArrays

include(joinpath("..","..","fem_example.jl"))

with_mpi(fem_example)

end # module

