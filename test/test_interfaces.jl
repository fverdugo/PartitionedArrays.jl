
using LinearAlgebra
using SparseArrays
using DistributedDataDraft
using Test

function test_interfaces(parts)

  nparts = num_parts(parts)
  @assert nparts == 4

  parts2 = get_part_ids(get_backend(parts),nparts)
  map_parts(parts,parts2) do part1, part2
    @test part1 == part2
  end

  parts_rcv = map_parts(parts) do part
    if part == 1
      [2,3]
    elseif part == 2
      [4,]
    elseif part == 3
      [1,2]
    else
      [1,3]
    end
  end
  
  parts_snd = map_parts(parts) do part
    if part == 1
      [3,4]
    elseif part == 2
      [1,3]
    elseif part == 3
      [1,4]
    else
      [2]
    end
  end
  
  data_snd = map_parts(i->10*i,parts_snd)
  data_rcv = map_parts(similar,parts_rcv)

  exchange!(
    data_rcv,
    data_snd,
    parts_rcv,
    parts_snd)

  map_parts(parts,data_rcv) do part, data_rcv
    if part == 1
      r = [10,10]
    elseif part == 2
      r = [20]
    elseif part == 3
      r = [30,30]
    else
      r= [40,40]
    end
    @test r == data_rcv
  end

  data_rcv, t = async_exchange(
    data_snd,
    parts_rcv,
    parts_snd)

  map_parts(schedule,t)
  map_parts(wait,t)

  map_parts(parts,data_rcv) do part, data_rcv
    if part == 1
      r = [10,10]
    elseif part == 2
      r = [20]
    elseif part == 3
      r = [30,30]
    else
      r= [40,40]
    end
    @test r == data_rcv
  end

  data_rcv = exchange(
    data_snd,
    parts_rcv,
    parts_snd)

  map_parts(parts,data_rcv) do part, data_rcv
    if part == 1
      r = [10,10]
    elseif part == 2
      r = [20]
    elseif part == 3
      r = [30,30]
    else
      r= [40,40]
    end
    @test r == data_rcv
  end

  parts_snd_2 = discover_parts_snd(parts_rcv)
  map_parts(parts_snd,parts_snd_2) do parts_snd, parts_snd_2
    @test parts_snd == parts_snd_2
  end

  n = 10

  lids = map_parts(parts) do part
    if part == 1
      IndexSet(part,n,[1,2,3,5,7,8],[1,1,1,2,3,3])
    elseif part == 2
      IndexSet(part,n,[2,4,5,10],[1,2,2,4])
    elseif part == 3
      IndexSet(part,n,[6,7,8,5,4,10],[3,3,3,2,2,4])
    else
      IndexSet(part,n,[1,3,7,9,10],[1,1,3,4,4])
    end
  end

  exchanger = Exchanger(lids)

  map_parts(parts,exchanger.parts_snd,exchanger.lids_snd) do part, parts_snd, lids_snd
    if part == 1
      _parts = [2,4]
      ids = [[2],[1,3]]
    elseif part == 2
      _parts = [1,3]
      ids = [[3],[3,2]]
    elseif part == 3
      _parts = [1,4]
      ids = [[2,3],[2]]
    else
      _parts = [2,3]
      ids = [[5],[5]]
    end
    @test _parts == parts_snd
    @test ids == lids_snd
  end

  values = map_parts(parts,lids) do part, lids
    values = fill(0.0,num_lids(lids))
    for lid in 1:length(lids.lid_to_part)
      owner = lids.lid_to_part[lid]
      if owner == part
        values[lid] = 10*part
      end
    end
    values
  end

  exchange!(values,exchanger)

  map_parts(values,lids) do values, lids
    for lid in 1:length(lids.lid_to_part)
      owner = lids.lid_to_part[lid]
      @test values[lid] == 10*owner
    end
  end

  exchanger_rcv = exchanger # receives data at ghost ids from remote parts
  exchanger_snd = reverse(exchanger_rcv) # sends data at ghost ids to remote parts

  values = map_parts(parts,lids) do part, lids
    values = fill(0.0,num_lids(lids))
    for lid in 1:length(lids.lid_to_part)
      values[lid] = 10*part
    end
    values
  end
  exchange!(values,exchanger_snd;reduce_op=+)
  exchange!(values,exchanger_rcv)

  ids = DistributedRange(n,lids)

  map_parts(ids.lids) do lids
    @test lids.ngids == n
  end
  @test num_parts(ids) == nparts
  @test num_gids(ids) == n

  ids2 = DistributedRange(parts,n)

  gids = map_parts(parts) do part
    if part == 1
      gids = [1,4,6]
    elseif part == 2
      gids = [3,1,2,8]
    elseif part == 3
      gids = [1,9,6]
    else
      gids = [3,2,8,10]
    end
  end

  ids3 = add_gid(ids2,gids)

  to_lid!(gids,ids3)
  to_gid!(gids,ids3)
  v = DistributedVector(gids,map_parts(copy,gids),ids3;ids=:global)
  v = DistributedVector(gids,map_parts(copy,gids),ids3;ids=:local)
  v = DistributedVector(gids,map_parts(copy,gids),n;ids=:global)
  u = 2*v
  map_parts(u.values,v.values) do u,v
    @test u == 2*v
  end
  u = v + u
  map_parts(u.values,v.values) do u,v
    @test u == 3*v
  end

  map_parts(parts,local_view(v)) do part,v
    if part == 3
      v[1] = 6
    end
  end

  map_parts(parts,local_view(v)) do part,v
    if part == 3
      @test v[1] == 6
    end
  end

  map_parts(parts,global_view(v)) do part,v
    if part == 4
      v[9] = 6
    end
  end

  map_parts(parts,global_view(v)) do part,v
    if part == 4
      @test v[9] == 6
    end
  end

  v = DistributedVector{Float64}(undef,ids)
  fill!(v,1.0)

  v = DistributedVector{Float64}(undef,ids)
  map_parts(parts,v.values,v.ids.lids) do part, values, lids
    for lid in 1:length(lids.lid_to_part)
      owner = lids.lid_to_part[lid]
      if owner == part
        values[lid] = 10*part
      end
    end
  end
  exchange!(v)
  map_parts(parts,v.values,v.ids.lids) do part, values, lids
    for lid in 1:length(lids.lid_to_part)
      owner = lids.lid_to_part[lid]
      @test values[lid] == 10*owner
    end
  end

  assemble!(v)

  I,J,V = map_parts(parts) do part
    if part == 1
      [1,2], [2,6], [1.0,2.0]
    elseif part == 2
      [3,3,4], [4,8,4], [1.0,2.0,3.0]
    elseif part == 3
      [5,5], [5,6], [1.0,2.0]
    else
      [9,9,8], [3,2,9], [1.0,2.0,3.0]
    end
  end
  A = DistributedSparseMatrix(I,J,V,n,n;ids=:global)
  local_view(A)
  global_view(A)

  x = DistributedVector{Float64}(undef,A.col_ids)
  fill!(x,1.0)
  y = A*x
  r = y - y

  col_ids = ids
  row_ids = col_ids

  values = map_parts(row_ids.lids,col_ids.lids) do row_ids, col_ids
    i = collect(1:num_lids(row_ids))
    j = i
    v = fill(2.0,length(i))
    sparse(i,j,v,num_lids(row_ids),num_lids(col_ids))
  end

  x = DistributedVector{Float64}(undef,col_ids)
  fill!(x,3)
  b = DistributedVector{Float64}(undef,row_ids)


  A = DistributedSparseMatrix(values,row_ids,col_ids)
  mul!(b,A,x)

  map_parts(b.values) do values
    @test all( values .== 6 )
  end

  exchange!(A)
  assemble!(A)

end

