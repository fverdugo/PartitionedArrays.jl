module SequentialBackendInterfacesTests

using PartitionedArrays

include(joinpath("..","interfaces_tests.jl"))
with_backend(interfaces_tests,SequentialBackend())

end # module
