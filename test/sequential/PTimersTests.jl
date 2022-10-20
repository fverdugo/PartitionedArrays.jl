module PTimersTests

include("../test_p_timers.jl")

nparts = 5
with_backend(test_p_timers,SequentialBackend(),nparts)

end # module
