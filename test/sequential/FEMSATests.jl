include("../test_fem_sa.jl")

nparts = (2,2)
prun(test_fem_sa,sequential,nparts)

nparts = 4
prun(test_fem_sa,sequential,nparts)

