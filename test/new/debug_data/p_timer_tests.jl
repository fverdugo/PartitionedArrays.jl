module DebugDataPTimerTests

using PartitionedArrays

include(joinpath("..","p_timer_tests.jl"))

with_debug_data(p_timer_tests)

end # module

