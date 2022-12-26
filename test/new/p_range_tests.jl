
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

function p_range_tests(distribute)

   rank = distribute(LinearIndices((4,)))

   np = (2,2)
   n = (10,10)
   ghost = (true,true)
   periodic = (true,false)
   pr = PRange(ConstantBlockSize(),rank,np,n)
   pr = PRange(ConstantBlockSize(),rank,np,n,ghost)
   pr = PRange(ConstantBlockSize(),rank,np,n,ghost,periodic)

   np = 4
   n = 100
   ghost = true
   periodic = true
   pr = PRange(ConstantBlockSize(),rank,np,n)
   pr = PRange(ConstantBlockSize(),rank,np,n,ghost)
   pr = PRange(ConstantBlockSize(),rank,np,n,ghost,periodic)

   n_own = map(rank) do rank
       mod(rank,3) + 2
   end

   pr = PRange(VariableBlockSize(),n_own)



   #add_ghost!(pr,gids,owners)


end
