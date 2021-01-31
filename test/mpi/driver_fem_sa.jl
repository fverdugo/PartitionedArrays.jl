include("../test_fem_sa.jl")

nparts = (2,2)
prun(test_fem_sa,mpi,nparts)
