include("../test_fdm.jl")
nparts = 4
distributed_run(test_fdm,mpi,nparts)

