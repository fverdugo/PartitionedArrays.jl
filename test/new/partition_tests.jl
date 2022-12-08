module PartitionTests

using PartitionedArrays
using Test

@test 1:3 == local_range(1,10,3)
@test 4:6 == local_range(2,10,3)
@test 7:10 == local_range(3,10,3)
@test 1:4 == local_range(1,10,3,true)
@test 3:7 == local_range(2,10,3,true)
@test 6:10 == local_range(3,10,3,true)
@test 0:4 == local_range(1,10,3,true,true)
@test 3:7 == local_range(2,10,3,true,true)
@test 6:11 == local_range(3,10,3,true,true)

end # module
