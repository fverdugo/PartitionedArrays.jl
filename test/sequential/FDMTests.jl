module FDMTests

include("../test_fdm.jl")

nparts = (2,2,2)
prun(test_fdm,sequential,nparts)

nparts = 4
prun(test_fdm,sequential,nparts)

end # module
