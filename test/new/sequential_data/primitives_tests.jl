module DebugDataPrimitivesTests

using PartitionedArrays

include(joinpath("..","primitives_tests.jl"))

with_debug_data(primitives_tests)

end # module
