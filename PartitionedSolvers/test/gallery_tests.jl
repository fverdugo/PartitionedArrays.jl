module GalleryTests

using PartitionedArrays
using PartitionedSolvers
using Test

function main(distribute)
    test_all(distribute,(4,))
    test_all(distribute,(2,2))
    test_all(distribute,(2,1,2))
end

function main(distribute,parts_per_dir)
    p = prod(parts_per_dir)
    ranks = distribute(LinearIndices((p,)))
    nodes_per_dir = map(i->2*i,parts_per_dir)
    args = laplace_matrix_fdm(nodes_per_dir,parts_per_dir,ranks)
    A = psparse(args...) |> fetch
    A = psparse(args...;assembled=true) |> fetch
end

with_debug(main)

end # module
