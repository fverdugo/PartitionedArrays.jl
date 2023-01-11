module SequentialDataPVectorTests

using PartitionedArrays

include(joinpath("..","p_vector_tests.jl"))

with_sequential_data(p_vector_tests)

end # module

