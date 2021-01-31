include("../test_p_timers.jl")
nparts = 4
prun(test_p_timers,mpi,nparts)
