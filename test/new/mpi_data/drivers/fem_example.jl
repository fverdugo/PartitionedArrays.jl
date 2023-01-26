module MPIDataFEMExample

using PartitionedArrays

include(joinpath("..","..","fem_example.jl"))

with_mpi_data(fem_example)

end # module

