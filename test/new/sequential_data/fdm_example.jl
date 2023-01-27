module DebugDataFDMExample

using PartitionedArrays

include(joinpath("..","fdm_example.jl"))

with_debug_data(fdm_example)

end # module

