module MPIDataFDMExample

using PartitionedArrays

include(joinpath("..","..","fdm_example.jl"))

with_mpi(fdm_example)

end # module

