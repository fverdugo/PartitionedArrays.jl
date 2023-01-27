module DebugDataFDMExample

using PartitionedArrays

include(joinpath("..","fdm_example.jl"))

with_debug(fdm_example)

end # module

