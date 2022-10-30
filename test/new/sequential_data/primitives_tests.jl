module SequentialDataPrimitivesTests

using PartitionedArrays

include(joinpath("..","primitives_tests.jl"))

primitives_tests(SequentialData)

end # module
