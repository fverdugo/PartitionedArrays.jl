module PartitionTests

using PartitionedArrays
using Test

@test 1:3 == local_range(1,3,10)
@test 4:6 == local_range(2,3,10)
@test 7:10 == local_range(3,3,10)
@test 1:4 == local_range(1,3,10,true)
@test 3:7 == local_range(2,3,10,true)
@test 6:10 == local_range(3,3,10,true)
@test 0:4 == local_range(1,3,10,true,true)
@test 3:7 == local_range(2,3,10,true,true)
@test 6:11 == local_range(3,3,10,true,true)

#partition = UniformBlockPartition((3,),(10,))
#@test size(partition) == (3,)
#@test partition[2] == CartesianIndices((4:6,))
#@test eltype(partition)  == typeof(partition[2])
#
#partition = UniformBlockPartition((3,3),(10,10))
#@test size(partition) == (3,3)
#@test partition[2,1] == CartesianIndices((4:6, 1:3))
#@test eltype(partition)  == typeof(partition[2,1])
#
#partition = BlockPartition(([1,3,7,11],[1,5,8,11]))
#@test size(partition) == (3,3)
#@test partition[2,1] == CartesianIndices((3:6, 1:4))
#@test eltype(partition)  == typeof(partition[2,1])
#
#partition = BlockPartition(([1,3,7,11],))
#@test size(partition) == (3,)
#@test partition[2] == CartesianIndices((3:6,))
#@test eltype(partition)  == typeof(partition[2])
#
#owner = part_owner(partition)


end # module
