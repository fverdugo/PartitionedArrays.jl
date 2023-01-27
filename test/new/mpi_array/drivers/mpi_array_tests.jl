module MPIDataTests

using PartitionedArrays

with_mpi() do distribute
    rank = distribute(LinearIndices((4,)))
    display(rank)
    rank = distribute(LinearIndices((2,2)))
    display(rank)
end

end # module
