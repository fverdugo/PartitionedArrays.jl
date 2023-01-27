module DebugDataPVectorTests

using PartitionedArrays

include(joinpath("..","p_vector_tests.jl"))

with_debug_data(p_vector_tests)

end # module

