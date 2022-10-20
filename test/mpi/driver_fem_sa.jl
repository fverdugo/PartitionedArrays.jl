include("../test_fem_sa.jl")

nparts = (2,2)
with_backend(test_fem_sa,MPIBackend(),nparts)
