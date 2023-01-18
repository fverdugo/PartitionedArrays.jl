module SequentialDataPSparseMatrixTests

using PartitionedArrays

include(joinpath("..","p_sparse_matrix_tests.jl"))

with_sequential_data(p_sparse_matrix_tests)

end # module

