module DebugDataPrimitivesTests

using PartitionedArrays

include(joinpath("..","primitives_tests.jl"))

with_debug(primitives_tests)

end # module
