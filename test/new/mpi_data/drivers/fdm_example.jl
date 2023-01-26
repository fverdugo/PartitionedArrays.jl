module MPIDataFDMExample

using PartitionedArrays

include(joinpath("..","..","fdm_example.jl"))

with_mpi_data(fdm_example)

end # module

