module SequentialDataPRangeTests

using PartitionedArrays

include(joinpath("..","p_range_tests.jl"))

with_sequential_data(p_range_tests)

end # module
