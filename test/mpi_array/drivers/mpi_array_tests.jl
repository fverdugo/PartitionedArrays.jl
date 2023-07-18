module MPIArrayTests

using PartitionedArrays
using Test

with_mpi() do distribute
    rank = distribute(LinearIndices((4,)))
    display(rank)
    rank = distribute(LinearIndices((2,2)))
    display(rank)

    n = 4
    row_partition = uniform_partition(rank,n)
    my_own_to_global = map(own_to_global,row_partition)
    ids = gather(my_own_to_global)
    map_main(ids) do myids
        @test myids == [[1],[2],[3],[4]]
    end
    ids = gather(my_own_to_global,destination=:all)
    map(ids) do myids
        @test myids == [[1],[2],[3],[4]]
    end

end

end # module
