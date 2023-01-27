module DebugArrayPVectorTests

using PartitionedArrays

include(joinpath("..","p_vector_tests.jl"))

with_debug(p_vector_tests)

end # module

