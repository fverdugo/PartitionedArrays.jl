module FDMTests

include("../test_fdm.jl")

nparts = (2,2,2)
with_backend(test_fdm,SequentialBackend(),nparts)

nparts = 4
with_backend(test_fdm,SequentialBackend(),nparts)

end # module
