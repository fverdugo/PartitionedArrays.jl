module FDMTests

include("../test_fdm.jl")

nparts = (2,2,2)
with_backend(test_fdm,sequential,nparts)

nparts = 4
with_backend(test_fdm,sequential,nparts)

end # module
