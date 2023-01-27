module DebugDataPRangeTests

using PartitionedArrays

include(joinpath("..","p_range_tests.jl"))

with_debug_data(p_range_tests)

end # module
