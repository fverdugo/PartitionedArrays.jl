module DebugDataFEMExample

using PartitionedArrays

include(joinpath("..","fem_example.jl"))

with_debug_data(fem_example)

end # module

