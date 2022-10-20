include("../test_fdm.jl")
nparts = (2,2,2)
with_backend(test_fdm,mpi,nparts)

