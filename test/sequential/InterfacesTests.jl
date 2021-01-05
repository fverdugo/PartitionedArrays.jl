module InterfacesTests

include("../test_interfaces.jl")

nparts = 4
distributed_run(test_interfaces,sequential,nparts)

nparts = (2,2)
distributed_run(test_interfaces,sequential,nparts)

end # module

