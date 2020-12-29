
include("test_interfaces.jl")

nparts = 4
distributed_run(test_interfaces,mpi,nparts)
