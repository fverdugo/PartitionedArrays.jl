module DebugArrayPRangeTests

using PartitionedArrays

include(joinpath("..","p_range_tests.jl"))

with_debug(p_range_tests)

end # module
