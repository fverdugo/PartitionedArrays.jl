module FDMTests

include("../test_fdm.jl")
nparts = 4
distributed_run(test_fdm,sequential,nparts)

end # module
