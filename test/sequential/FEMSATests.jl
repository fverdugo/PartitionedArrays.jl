include("../test_fem_sa.jl")

nparts = (2,2)
distributed_run(test_fem_sa,sequential,nparts)

nparts = 4
distributed_run(test_fem_sa,sequential,nparts)

