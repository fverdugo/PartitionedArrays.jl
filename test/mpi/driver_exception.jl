include("../test_exception.jl")
nparts = (2,2,2)
with_backend(test_exception,mpi,nparts)
