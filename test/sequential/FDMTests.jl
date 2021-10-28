module FDMTests

include("../test_fdm.jl")

nparts = (2,2,2)
prun_debug(test_fdm,sequential,nparts)

nparts = 4
prun_debug(test_fdm,sequential,nparts)

end # module
