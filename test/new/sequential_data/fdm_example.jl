module SequentialDataPVectorTests

using PartitionedArrays

include(joinpath("..","fdm_example.jl"))

with_sequential_data(fdm_example)

end # module

