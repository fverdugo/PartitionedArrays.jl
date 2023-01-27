module DebugDataFEMExample

using PartitionedArrays

include(joinpath("..","fem_example.jl"))

with_debug(fem_example)

end # module

