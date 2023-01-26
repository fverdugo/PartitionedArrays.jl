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
    p = emit(p_main)
    map(parts,p) do part,part_fail
        @assert  part_fail != part
    end
end

with_mpi_data(exception_tests)


