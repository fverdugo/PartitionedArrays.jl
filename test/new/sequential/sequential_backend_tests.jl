module SequentialBackendTests

using PartitionedArrays
using Test

PartitionedArrays.scalar_indexing(:error)

r = with_backend(SequentialBackend()) do backend

    parts = cartesian_indices(backend,(3,))
    display(parts)

    @test length(parts) == 3

    @test_throws ErrorException("Scalar indexing on SequentialArray is not allowed for performance reasons.") parts[end]

    parts = linear_indices(backend,(3,4))
    display(parts)

    @test length(parts) == 12

    a = map(-,parts)
    map(a,parts) do a,part
        @test a == -part
    end

    -1
end

@test r == -1

end # module
