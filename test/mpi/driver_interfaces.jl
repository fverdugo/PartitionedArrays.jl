include("../test_interfaces.jl")
nparts = 4
with_backend(test_interfaces,mpi,nparts)
