
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
   uniform_partition(rank,n)

   # Uniform linear partition with one layer of ghost
   ghost = true
   uniform_partition(rank,n,ghost)

   # Uniform linear partition with one layer of ghost
   # and periodic ghost
   periodic = true
   uniform_partition(rank,n,ghost,periodic)

   # uniform Cartesian partition without ghost
   np = (2,2)
   n = (10,10)
   uniform_partition(rank,np,n)

   # uniform Cartesian partition with one layer of ghost
   # in the selected directions
   np = (2,2)
   n = (10,10)
   ghost = (true,true)
   uniform_partition(rank,np,n,ghost)

   # uniform Cartesian partition with one layer of ghost
   # in the selected directions
   np = (2,2)
   n = (10,10)
   periodic = (true,true)
   uniform_partition(rank,np,n,ghost,periodic)

   # Custom linear partition with no ghost
   n_own = map(rank) do rank
       mod(rank,3) + 2
   end
   n=sum(n_own)
   variable_partition(n_own,n)

   # Custom linear partition with no ghost
   # scan to find the first id in each block is done by the caller
   start = scan(+,n_own,type=:exclusive,init=1)
   variable_partition(n_own,n;start)

   # Custom linear partition with arbitrary ghost
   # Here the ghost need to be non-repeated and actual ghost values
   # This requires a lot of communication to find
   # the owner of each given gid
   gids = map(rank) do rank
       Int[]
   end
   # First create a partition without ghost
   partition_without_ghost = variable_partition(n_own,n;start)
   # Then replace the ghost
   owners = find_owner(partition_without_ghost,gids)
   partition_with_ghost = map(replace_ghost,partition_without_ghost,gids,owners)

   # Custom linear partition with ghost
   # Here the gids can be whatever
   # Only the ghost not already present will be added
   # This requires a lot of communication to find
   # the owner of each given gid
   partition_without_ghost = variable_partition(n_own,n;start)
   owners = find_owner(partition_with_ghost,gids)
   partition_with_ghost = map(union_ghost,partition_without_ghost,gids,owners)

   # Custom general partition by providing
   # info about the local indices
   # We fill with a uniform partition as an example
   np = (2,2)
   n = (10,10)
   ghost = (true,true)
   n_global = prod(n)
   tentative_partition = uniform_partition(rank,np,n,ghost)
   arbitrary_partition = map(tentative_partition) do old_local_indices
       local_to_global = get_local_to_global(old_local_indices) |> collect
       local_to_owner = get_local_to_owner(old_local_indices) |> collect
       owner  = get_owner(old_local_indices)
       LocalIndices(n_global,owner,local_to_global,local_to_owner)
   end

   # Custom general partition by providing
   # info about the own and ghost indices
   # local indices are defined by concatenating
   # own and ghost
   custom_partition = map(tentative_partition) do old_local_indices
       owner = get_owner(old_local_indices)
       own_to_global = get_own_to_global(old_local_indices) |> collect
       ghost_to_global = get_ghost_to_global(old_local_indices) |> collect
       ghost_to_owner = get_ghost_to_owner(old_local_indices) |> collect
       own = OwnIndices(n_global,owner,own_to_global)
       ghost = GhostIndices(n_global,ghost_to_global,ghost_to_owner)
       OwnAndGhostIndices(own,ghost)
   end

   # Custom general partition by providing
   # info about the own and ghost indices
   # local indices are defined by concatenating
   # own and ghost plus an arbitrary permutation
   custom_partition = map(tentative_partition) do old_local_indices
       owner = get_owner(old_local_indices)
       own_to_global = get_own_to_global(old_local_indices) |> collect
       ghost_to_global = get_ghost_to_global(old_local_indices) |> collect
       ghost_to_owner = get_ghost_to_owner(old_local_indices) |> collect
       own = OwnIndices(n_global,owner,own_to_global)
       ghost = GhostIndices(n_global,ghost_to_global,ghost_to_owner)
       n_local = length(get_local_to_global(old_local_indices))
       perm = collect(n_local:-1:1)
       permute_indices(OwnAndGhostIndices(own,ghost),perm)
   end

   n = 10
   parts = rank
   ids = map(parts) do part
       if part == 1
           LocalIndices(n,part,[1,2,3,5,7,8],Int32[1,1,1,2,3,3])
       elseif part == 2
           LocalIndices(n,part,[2,4,5,10],Int32[1,2,2,4])
       elseif part == 3
           LocalIndices(n,part,[6,7,8,5,4,10],Int32[3,3,3,2,2,4])
       else
           LocalIndices(n,part,[1,3,7,9,10],Int32[1,1,3,4,4])
       end
   end

   ids = uniform_partition(parts,n)
   @test length(ids) == length(parts)

   gids = map(parts) do part
       if part == 1
           [1,4,6]
       elseif part == 2
           [3,1,2,8]
       elseif part == 3
           [1,9,6]
       else
           [3,2,8,10]
       end
   end

   owners = find_owner(ids,gids)
   ids3 = map(union_ghost,ids,gids,owners)
   map(to_local!,gids,ids3)
   map(to_global!,gids,ids3)

   a = map(parts) do part
       if part == 1
           4
       elseif part == 2
           2
       elseif part == 3
           6
       else
           3
       end
   end
   ids5 = variable_partition(a,sum(a))
   map(parts,ids5) do part, local_to_global
       if part == 1
           @test local_to_global == [1, 2, 3, 4]
       elseif part == 2
           @test local_to_global == [5, 6]
       elseif part == 3
           @test local_to_global == [7, 8, 9, 10, 11, 12]
       else
           @test local_to_global == [13, 14, 15]
       end
   end

   ids4 = uniform_partition(parts,(2,2),(5,4))
   @test length(PRange(ids4)) == 4*5
   map(parts,ids4) do part, lid_to_gid
       if part == 1
           @test lid_to_gid == [1, 2, 6, 7]
       elseif part == 2
           @test lid_to_gid == [3, 4, 5, 8, 9, 10]
       elseif part == 3
           @test lid_to_gid == [11, 12, 16, 17]
       else
           @test lid_to_gid == [13, 14, 15, 18, 19, 20]
       end
   end

   ids4 = uniform_partition(parts,(2,2),(5,4),(false,false))
   ids4 = uniform_partition(parts,(2,2),(5,4),(true,true))
   map(parts,ids4) do part, lid_to_gid
       if part == 1
           @test lid_to_gid == [1, 2, 3, 6, 7, 8, 11, 12, 13]
       elseif part == 2
           @test lid_to_gid == [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15]
       elseif part == 3
           @test lid_to_gid == [6, 7, 8, 11, 12, 13, 16, 17, 18]
       else
           @test lid_to_gid == [7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20]
       end
   end

   ids4 = uniform_partition(parts,(2,2),(4,4),(true,true),(true,true))
   map(parts,ids4) do part,lid_to_gid
       if part == 1
           @test lid_to_gid == [16, 13, 14, 15, 4, 1, 2, 3, 8, 5, 6, 7, 12, 9, 10, 11]
       elseif part ==2
           @test lid_to_gid == [14, 15, 16, 13, 2, 3, 4, 1, 6, 7, 8, 5, 10, 11, 12, 9]
       elseif part == 3
           @test lid_to_gid == [8, 5, 6, 7, 12, 9, 10, 11, 16, 13, 14, 15, 4, 1, 2, 3]
       else
           @test lid_to_gid == [6, 7, 8, 5, 10, 11, 12, 9, 14, 15, 16, 13, 2, 3, 4, 1]
       end
   end

   ids4 = uniform_partition(parts,(2,2),(4,4),(true,true),(false,true))
   map(parts,ids4) do part,lid_to_gid
       if part == 1
           @test lid_to_gid == [13, 14, 15, 1, 2, 3, 5, 6, 7, 9, 10, 11]
       elseif part ==2
           @test lid_to_gid == [14, 15, 16, 2, 3, 4, 6, 7, 8, 10, 11, 12]
       elseif part == 3
           @test lid_to_gid == [5, 6, 7, 9, 10, 11, 13, 14, 15, 1, 2, 3]
       else
           @test lid_to_gid == [6, 7, 8, 10, 11, 12, 14, 15, 16, 2, 3, 4]
       end
   end

   display(PRange(ids4))

end
