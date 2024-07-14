using PartitionedArrays
using PSparseMatrixGallery

function test_all(distribute)
    test_all(distribute,(4,))
    test_all(distribute,(2,2))
    test_all(distribute,(2,1,2))
end

function test_all(distribute,parts_per_dir)

    p = prod(parts_per_dir)
    ranks = distribute(LinearIndices((p,)))
    nodes_per_dir = map(i->2*i,parts_per_dir)
    args = laplace_matrix_fdm(;nodes_per_dir,parts_per_dir,ranks)
    A = psparse(args...) |> fetch
    display(A)
    display(centralize(A))

end
