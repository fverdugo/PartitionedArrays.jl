include("../test_fdm.jl")
nparts = (2,2,2)
prun(test_fdm,mpi,nparts)

