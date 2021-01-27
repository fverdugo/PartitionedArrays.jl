include("../test_p_timers.jl")
nparts = 4
distributed_run(test_p_timers,mpi,nparts)
