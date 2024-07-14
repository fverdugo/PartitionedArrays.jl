
using PartitionedArrays
using Test

function gallery_tests(distribute)
    gallery_tests(distribute,(4,))
    gallery_tests(distribute,(2,2))
    gallery_tests(distribute,(2,1,2))
end

function gallery_tests(distribute,parts_per_dir)
    p = prod(parts_per_dir)
    ranks = distribute(LinearIndices((p,)))
    nodes_per_dir = map(i->2*i,parts_per_dir)
    args = laplace_matrix_fdm(nodes_per_dir,parts_per_dir,ranks)
    A = psparse(args...) |> fetch
    A = psparse(args...;assembled=true) |> fetch
end


