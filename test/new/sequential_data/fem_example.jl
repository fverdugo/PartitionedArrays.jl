module SequentialDataPVectorTests

using PartitionedArrays

include(joinpath("..","fem_example.jl"))

with_sequential_data(fem_example)

end # module

