include("../test_p_timers.jl")
nparts = 4
with_backend(test_p_timers,mpi,nparts)
