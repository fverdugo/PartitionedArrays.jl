module SequentialDataPrimitivesTests

using PartitionedArrays

include(joinpath("..","primitives_tests.jl"))

with_sequential_data(primitives_tests)

end # module
