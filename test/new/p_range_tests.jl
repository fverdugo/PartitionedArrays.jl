
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


   # Uniform linear partition without ghost
   np = 4
   n = 100
   pr = PRange(ConstantBlockSize(),rank,np,n)

   # Uniform linear partition with one layer of ghost
   ghost = true
   pr = PRange(ConstantBlockSize(),rank,np,n,ghost)

   # Uniform linear partition with one layer of ghost
   # and periodic ghost
   periodic = true
   pr = PRange(ConstantBlockSize(),rank,np,n,ghost,periodic)

   # uniform Cartesian partition without ghost
   np = (2,2)
   n = (10,10)
   pr = PRange(ConstantBlockSize(),rank,np,n)

   # uniform Cartesian partition with one layer of ghost
   # in the selected directions
   np = (2,2)
   n = (10,10)
   ghost = (true,true)
   pr = PRange(ConstantBlockSize(),rank,np,n,ghost)

   # uniform Cartesian partition with one layer of ghost
   # in the selected directions
   np = (2,2)
   n = (10,10)
   periodic = (true,true)
   pr = PRange(ConstantBlockSize(),rank,np,n,ghost,periodic)

   # Custom linear partition with no ghost
   n_own = map(rank) do rank
       mod(rank,3) + 2
   end
   pr = PRange(VariableBlockSize(),n_own)

   # Custom linear partition with no ghost
   # reduction to discover ngids is done by the caller
   n = sum(n_own)
   pr = PRange(VariableBlockSize(),n_own,n)

   # Custom linear partition with no ghost
   # reduction to discover ngids is done by the caller
   # scan to find the first id in each block is done by the caller
   start = scan(+,n_own,type=:exclusive,init=1)
   pr = PRange(VariableBlockSize(),n_own,n,start)

   # Custom linear partition with ghost
   # Here the ghost need to be non-repeated actual ghost values
   # This requires a lot of communication to find
   # the owner of each given gid
   gids = map(rank) do rank
       Int[]
   end
   pr = PRange(VariableBlockSize(),n_own,n,start,gids)

   # Same as before but save some communications
   # by providing the owners
   owners = map(rank) do rank
       Int32[]
   end
   pr = PRange(VariableBlockSize(),n_own,n,start,gids,owners)

   # We can achieve the same without taking ownership of the
   # gids and owners
   pr = PRange(VariableBlockSize(),n_own)
   append_ghost!(pr,gids)

   # Same as before but save some communications
   # by providing the owners
   pr = PRange(VariableBlockSize(),n_own)
   append_ghost!(pr,gids,owners)

   # Custom linear partition with ghost
   # Here the gids can be whatever
   # Only the ghost not already present will be added
   # This requires a lot of communication to find
   # the owner of each given gid
   pr = PRange(VariableBlockSize(),n_own)
   union_ghost!(pr,gids)

   # Same as before but save some communications
   # by providing the owners
   pr = PRange(VariableBlockSize(),n_own)
   union_ghost!(pr,gids,owners)

   # Custom general partition by providing
   # info about the local indices
   # We fill with a uniform partition as an example
   n = 10
   np = length(rank)
   ghost = true
   local_indices = map(rank) do rank
       local_to_global = collect(local_range(rank,np,n,ghost))
       n_local = length(local_to_global)
       local_to_owner = fill(Int32(rank),n_local)
       o = bounday_owner(rank,np,n,ghost)
       local_to_owner[1] = o[1]
       local_to_owner[end] = o[end]
       LocalIndices(n,rank,local_to_global,local_to_owner)
   end
   pr = PRange(n,local_indices)

   # Custom general partition by providing
   # info about the own and ghost indices
   # local indices are defined by concatenating
   # own and ghost
   n = 10
   np = length(rank)
   ghost = true
   local_indices = map(rank) do rank
       local_to_global = collect(local_range(rank,np,n,ghost))
       n_local = length(local_to_global)
       local_to_owner = fill(Int32(rank),n_local)
       o = bounday_owner(rank,np,n,ghost)
       local_to_owner[1] = o[1]
       local_to_owner[end] = o[end]
       is_own = local_to_owner .== rank
       own_to_global = local_to_global[is_own]
       own = OwnIndices(n,rank,own_to_global)
       is_ghost = local_to_global .!= rank
       ghost_to_global = local_to_global[is_ghost]
       ghost_to_owner = local_to_owner[is_ghost]
       _ghost = GhostIndices(n,ghost_to_global,ghost_to_owner)
       OwnAndGhostIndices(own,_ghost)
   end
   pr = PRange(n,local_indices)

   

   



end
