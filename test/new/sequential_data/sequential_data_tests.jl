module SequentialDataTests

using PartitionedArrays
using Test

distribute(x) = SequentialData(x)

parts = distribute(CartesianIndices((3,)))
display(parts)

@test length(parts) == 3

msg = "Scalar indexing on SequentialData is not allowed for performance reasons."

@test_throws ErrorException(msg) parts[end]

parts = distribute(LinearIndices((3,4)))
display(parts)

@test length(parts) == 12

a = map(-,parts)
map(a,parts) do a,part
    @test a == -part
end

b = similar(a)
@test typeof(a) == typeof(b)


end # module
