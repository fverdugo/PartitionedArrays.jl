module InterfacesTests

using DistributedDataDraft
using Test

include("test_interfaces.jl")

nparts = 4
distributed_run(test_interfaces,sequential,nparts)

end # module
