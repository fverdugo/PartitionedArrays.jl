module FDMTests

include("../test_fdm.jl")

nparts = (2,2,2)
distributed_run(test_fdm,sequential,nparts)

nparts = 4
distributed_run(test_fdm,sequential,nparts)

end # module
