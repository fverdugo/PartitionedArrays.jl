module SequentialDataPTimerTests

using PartitionedArrays

include(joinpath("..","p_timer_tests.jl"))

with_sequential_data(p_timer_tests)

end # module

