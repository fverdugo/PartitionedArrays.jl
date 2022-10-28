module SequentialArrayPrimitivesTests

using PartitionedArrays

include(joinpath("..","primitives_tests.jl"))

primitives_tests(SequentialArray)

end # module
