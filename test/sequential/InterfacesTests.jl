module InterfacesTests

include("../test_interfaces.jl")

nparts = 4
prun(test_interfaces,sequential,nparts)

nparts = (2,2)
prun(test_interfaces,sequential,nparts)

end # module

