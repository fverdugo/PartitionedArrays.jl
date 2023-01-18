
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
   n = 10
   pr = uniform_partition(rank,n)

   # Uniform linear partition with one layer of ghost
   ghost = true
   pr = uniform_partition(rank,n,ghost)

   # Uniform linear partition with one layer of ghost
   # and periodic ghost
   periodic = true
   pr = uniform_partition(rank,n,ghost,periodic)

   # uniform Cartesian partition without ghost
   np = (2,2)
   n = (10,10)
   pr = uniform_partition(rank,np,n)

   # uniform Cartesian partition with one layer of ghost
   # in the selected directions
   np = (2,2)
   n = (10,10)
   ghost = (true,true)
   pr = uniform_partition(rank,np,n,ghost)

   # uniform Cartesian partition with one layer of ghost
   # in the selected directions
   np = (2,2)
   n = (10,10)
   periodic = (true,true)
   pr = uniform_partition(rank,np,n,ghost,periodic)

   # Custom linear partition with no ghost
   n_own = map(rank) do rank
       mod(rank,3) + 2
   end
   n=sum(n_own)
   pr = variable_partition(n_own,n)

   # Custom linear partition with no ghost
   # scan to find the first id in each block is done by the caller
   start = scan(+,n_own,type=:exclusive,init=1)
   pr = variable_partition(n_own,n;start)

   # Custom linear partition with arbitrary ghost
   # Here the ghost need to be non-repeated and actual ghost values
   # This requires a lot of communication to find
   # the owner of each given gid
   gids = map(rank) do rank
       Int[]
   end
   # First create a PRange without ghost
   pr = variable_partition(n_own,n;start)
   # Then replace the ghost
   pr = replace_ghost(pr,gids)

   # Same as before but save some communications
   # by providing the owners
   owners = map(rank) do rank
       Int32[]
   end
   pr = variable_partition(n_own,n;start)
   pr = replace_ghost(pr,gids,owners)

   # Custom linear partition with ghost
   # Here the gids can be whatever
   # Only the ghost not already present will be added
   # This requires a lot of communication to find
   # the owner of each given gid
   pr = variable_partition(n_own,n;start)
   pr = union_ghost(pr,gids)

   # Same as before but save some communications
   # by providing the owners
   pr = variable_partition(n_own,n;start)
   pr = union_ghost(pr,gids,owners)

   # Custom general partition by providing
   # info about the local indices
   # We fill with a uniform partition as an example
   np = (2,2)
   n = (10,10)
   ghost = (true,true)
   n_global = prod(n)
   old_pr = uniform_partition(rank,np,n,ghost)
   indices = map(old_pr.indices) do old_local_indices
       local_to_global = get_local_to_global(old_local_indices) |> collect
       local_to_owner = get_local_to_owner(old_local_indices) |> collect
       owner  = get_owner(old_local_indices)
       LocalIndices(n_global,owner,local_to_global,local_to_owner)
   end
   pr = PRange(n_global,indices)

   # Custom general partition by providing
   # info about the own and ghost indices
   # local indices are defined by concatenating
   # own and ghost
   indices = map(old_pr.indices) do old_local_indices
       owner = get_owner(old_local_indices)
       own_to_global = get_own_to_global(old_local_indices) |> collect
       ghost_to_global = get_ghost_to_global(old_local_indices) |> collect
       ghost_to_owner = get_ghost_to_owner(old_local_indices) |> collect
       own = OwnIndices(n_global,owner,own_to_global)
       ghost = GhostIndices(n_global,ghost_to_global,ghost_to_owner)
       OwnAndGhostIndices(own,ghost)
   end
   pr = PRange(n_global,indices)

   # Custom general partition by providing
   # info about the own and ghost indices
   # local indices are defined by concatenating
   # own and ghost plus an arbitrary permutation
   indices = map(old_pr.indices) do old_local_indices
       owner = get_owner(old_local_indices)
       own_to_global = get_own_to_global(old_local_indices) |> collect
       ghost_to_global = get_ghost_to_global(old_local_indices) |> collect
       ghost_to_owner = get_ghost_to_owner(old_local_indices) |> collect
       own = OwnIndices(n_global,owner,own_to_global)
       ghost = GhostIndices(n_global,ghost_to_global,ghost_to_owner)
       n_local = length(get_local_to_global(old_local_indices))
       perm = collect(n_local:-1:1)
       PermutedLocalIndices(OwnAndGhostIndices(own,ghost),perm)
   end
   pr = PRange(n_global,indices)

   parts = rank
   nparts = length(parts)
   @assert nparts == 4

   parts2 = linear_indices(parts)
   map(parts,parts2) do part1, part2
       @test part1 == part2
   end


end
