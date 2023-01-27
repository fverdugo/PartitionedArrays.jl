module DebugDataPSparseMatrixTests

using PartitionedArrays

include(joinpath("..","p_sparse_matrix_tests.jl"))

with_debug_data(p_sparse_matrix_tests)

end # module

