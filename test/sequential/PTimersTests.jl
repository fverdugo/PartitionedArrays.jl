module PTimersTests

include("../test_p_timers.jl")

nparts = 5
prun(test_p_timers,sequential,nparts)

end # module
