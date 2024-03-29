using PartitionedArrays

function exception_tests(distribute)
    parts = distribute(LinearIndices((4,)))
    nparts=length(parts)
    p_main = map(parts) do part
        if part == MAIN
            part_fail = rand(1:nparts)
        else
            0
        end
    end
    p = multicast(p_main)
    map(parts,p) do part,part_fail
        @assert  part_fail != part
    end
end

with_mpi(exception_tests)


