module SequentialBackendInterfacesTests

using PartitionedArrays

include(joinpath("..","interfaces_tests.jl"))

interfaces_tests(SequentialArray)

end # module
