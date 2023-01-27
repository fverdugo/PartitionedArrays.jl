module MPIDataPSparseMatrixTests

using PartitionedArrays

include(joinpath("..","..","p_sparse_matrix_tests.jl"))

with_mpi(p_sparse_matrix_tests)

end # module

