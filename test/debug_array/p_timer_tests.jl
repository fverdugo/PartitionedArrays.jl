module DebugArrayPTimerTests

using PartitionedArrays

include(joinpath("..","p_timer_tests.jl"))

with_debug(p_timer_tests)

end # module

