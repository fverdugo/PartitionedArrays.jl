using PartitionedArrays
using Test
using LinearAlgebra
using Random
using Distances

function p_sparse_matrix_tests(distribute)

    np = 4
    rank = distribute(LinearIndices((np,)))
    n = 10
    rows = uniform_partition(rank,n)
    I,J,V = map(rank) do rank
        if rank == 1
            I = [1,2,6,4]
            J = [1,4,2,7]
        elseif rank == 2
            I = [3,2]
            J = [4,7]
        elseif rank == 3
            I = [3,7,8,2]
            J = [7,3,8,6]
        else
            I = [6,9,10]
            J = [1,10,9]
        end
        I,J,fill(Float64(rank),length(J))
    end |> unpack

    rows = uniform_partition(rank,n)
    cols = rows
    A = psparse!(I,J,V,rows,cols) |> fetch

    

end

