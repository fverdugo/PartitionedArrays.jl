module MPIDataPSparseMatrixTests

using PartitionedArrays

include(joinpath("..","..","p_sparse_matrix_tests.jl"))

with_mpi_data(p_sparse_matrix_tests)

end # module

