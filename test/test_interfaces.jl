
using LinearAlgebra
using SparseArrays
using PartitionedArrays
using Test
using IterativeSolvers

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

  a = reduce_main(+,parts,init=0)
  @test get_main_part(a) == 1+2+3+4
  b = reduce_all(+,parts,init=0)
  map_parts(b) do b
    @test b == 1+2+3+4
  end
  @test reduce(+,parts,init=0) == 1+2+3+4
  @test sum(parts) == 1+2+3+4

  a = map_parts(parts) do part
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
  b = iscan(+,a,init=0)
  map_parts(parts,b) do part,b
    if part == 1
      @test b == 4
    elseif part == 2
      @test b == 6
    elseif part == 3
      @test b == 12
    else
      @test b == 15
    end
  end
  b, n = iscan(+,reduce,a,init=0)
  @test n == 15
  b, n = iscan_all(+,reduce,a,init=0)
  @test n == 15
  map_parts(b) do b
    @test b == [4,6,12,15]
  end

  b = xscan(+,a,init=1)
  map_parts(parts,b) do part,b
    if part == 1
      @test b == 1
    elseif part == 2
      @test b == 5
    elseif part == 3
      @test b == 7
    else
      @test b == 13
    end
  end

  b, n = xscan(+,reduce,a,init=1)
  @test n == 15+1
  b, n = xscan_all(+,reduce,a,init=1)
  @test n == 15+1
  map_parts(b) do b
    @test b == [1,5,7,13]
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

  partition = map_parts(parts) do part
    if part == 1
      IndexSet(part,[1,2,3,5,7,8],Int32[1,1,1,2,3,3])
    elseif part == 2
      IndexSet(part,[2,4,5,10],Int32[1,2,2,4])
    elseif part == 3
      IndexSet(part,[6,7,8,5,4,10],Int32[3,3,3,2,2,4])
    else
      IndexSet(part,[1,3,7,9,10],Int32[1,1,3,4,4])
    end
  end

  exchanger = Exchanger(partition)

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

  values = map_parts(parts,partition) do part, partition
    values = fill(0.0,num_lids(partition))
    for lid in 1:length(partition.lid_to_part)
      owner = partition.lid_to_part[lid]
      if owner == part
        values[lid] = 10*part
      end
    end
    values
  end

  exchange!(values,exchanger)

  map_parts(values,partition) do values, partition
    for lid in 1:length(partition.lid_to_part)
      owner = partition.lid_to_part[lid]
      @test values[lid] == 10*owner
    end
  end

  values_rcv = map_parts(values) do values
    fill(10.0,length(values))
  end

  values_snd = map_parts(values) do values
    fill(20.0,length(values))
  end

  exchange!(values_rcv,values_snd,exchanger)

  map_parts(values_rcv,partition) do values_rcv, partition
    for lid in 1:length(partition.lid_to_part)
      if partition.lid_to_part[lid] == partition.part
        @test values_rcv[lid] == 10.0
      else
        @test values_rcv[lid] == 20.0
      end
    end
  end

  map_parts(values_snd) do values_snd
    @test all(values_snd .== 20.0)
  end

  values = map_parts(parts,partition) do part, partition
    values = [ zeros(Int,3) for lid in 1:num_lids(partition)]
    for lid in partition.oid_to_lid
      gid = partition.lid_to_gid[lid]
      for i in 1:3
        values[lid][i] = 100*part + 10*gid + i
      end
    end
    Table(values)
  end

  exchange!(values,exchanger)

  map_parts(parts,partition,values) do part, partition, values
    for lid in 1:num_lids(partition)
      gid = partition.lid_to_gid[lid]
      owner = partition.lid_to_part[lid]
      for i in 1:3
        @test values[lid][i] == 100*owner + 10*gid + i
      end
    end
  end

  exchanger_rcv = exchanger # receives data at ghost ids from remote parts
  exchanger_snd = reverse(exchanger_rcv) # sends data at ghost ids to remote parts

  values = map_parts(parts,partition) do part, partition
    values = fill(0.0,num_lids(partition))
    for lid in 1:length(partition.lid_to_part)
      values[lid] = 10*part
    end
    values
  end
  exchange!(+,values,exchanger_snd)
  exchange!(values,exchanger_rcv)

  ids = PRange(n,partition)
  ids2 = copy(ids)

  @test ids !== ids2
  @test ids.partition !== ids2.partition
  @test ids.exchanger !== ids2.exchanger

  map_parts(ids.partition) do partition
    @test get_part(partition) == partition.part
    @test get_lid_to_gid(partition) == partition.lid_to_gid
    @test get_lid_to_part(partition) == partition.lid_to_part
    @test get_oid_to_lid(partition) == partition.oid_to_lid
    @test get_hid_to_lid(partition) == partition.hid_to_lid
    @test get_lid_to_ohid(partition) == partition.lid_to_ohid
    @test get_gid_to_lid(partition) == partition.gid_to_lid
  end
  @test num_parts(ids) == nparts
  @test num_gids(ids) == n

  ids2 = PRange(parts,n)
  @test ids2.ghost == false

  gids = map_parts(parts) do part
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

  i_to_part = map_parts(getindex,ids2.gid_to_part,gids)
  ids3 = add_gids(ids2,gids,i_to_part)
  @test ids3.ghost == true

  gids2 = map_parts(parts) do part
    if part == 1
      [4,6]
    elseif part == 2
      [1,2]
    elseif part == 3
      [1,9]
    else
      [3]
    end
  end

  hids = touched_hids(ids3,gids2)
  map_parts(hids,gids2,ids3.partition) do hids, gids, ids3
    @test gids == ids3.lid_to_gid[ids3.hid_to_lid[hids]]
  end

  ids3 = add_gids(ids2,gids)
  @test ids3.ghost == true
  to_lids!(gids,ids3)
  to_gids!(gids,ids3)

  a = map_parts(parts) do part
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
  ids5 = PRange(parts,a)
  map_parts(parts,ids5.partition,ids5.gid_to_part) do part, ids5, gid_to_part
    if part == 1
      @test ids5.lid_to_gid == [1, 2, 3, 4]
    elseif part == 2
      @test ids5.lid_to_gid == [5, 6]
    elseif part == 3
      @test ids5.lid_to_gid == [7, 8, 9, 10, 11, 12]
    else
      @test ids5.lid_to_gid == [13, 14, 15]
    end
    @test gid_to_part == [1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4]
  end
  ids5 = PRange(parts,reduce(+,a,init=0),a)
  ids5 = PRange(parts,reduce(+,a,init=0),a,xscan(+,a,init=1))

  hid_to_gid, hid_to_part = map_parts(parts) do part
    Int[], Int32[]
  end
  ids5 = PRange(parts,reduce(+,a,init=0),a,xscan(+,a,init=1),hid_to_gid,hid_to_part)

  if ndims(parts) > 1

    ids4 = PRange(parts,(5,4))
    @test ids4.ghost == false
    @test num_gids(ids4) == 4*5
    map_parts(parts,ids4.partition,ids4.gid_to_part) do part, ids4, gid_to_part
      if part == 1
        @test ids4.lid_to_gid == [1, 2, 6, 7]
      elseif part == 2
        @test ids4.lid_to_gid == [3, 4, 5, 8, 9, 10]
      elseif part == 3
        @test ids4.lid_to_gid == [11, 12, 16, 17]
      else
        @test ids4.lid_to_gid == [13, 14, 15, 18, 19, 20]
      end
      @test gid_to_part == [1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 3, 3, 4, 4, 4]
    end

    pcis = PCartesianIndices(parts,(5,4))
    map_parts(parts,pcis) do part, cis
      if part == 1
        @test cis == CartesianIndices((1:2,1:2))
      elseif part == 2
        @test cis == CartesianIndices((3:5,1:2))
      elseif part == 3
        @test cis == CartesianIndices((1:2,3:4))
      else
        @test cis == CartesianIndices((3:5,3:4))
      end
    end

    pcis = PCartesianIndices(parts,(5,4),no_ghost)
    pcis = PCartesianIndices(parts,(5,4),with_ghost)
    map_parts(parts,pcis) do part, cis
      if part == 1
        @test cis == CartesianIndices((1:3,1:3))
      elseif part == 2
        @test cis == CartesianIndices((2:5,1:3))
      elseif part == 3
        @test cis == CartesianIndices((1:3,2:4))
      else
        @test cis == CartesianIndices((2:5,2:4))
      end
    end

    ids4 = PRange(parts,(5,4),no_ghost)
    ids4 = PRange(parts,(5,4),with_ghost)
    map_parts(parts,ids4.partition,ids4.gid_to_part) do part, ids4,gid_to_part
      if part == 1
        @test ids4.lid_to_gid == [1, 2, 3, 6, 7, 8, 11, 12, 13]
      elseif part == 2
        @test ids4.lid_to_gid == [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15]
      elseif part == 3
        @test ids4.lid_to_gid == [6, 7, 8, 11, 12, 13, 16, 17, 18]
      else
        @test ids4.lid_to_gid == [7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20]
      end
      @test gid_to_part == [1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 3, 3, 4, 4, 4]
    end
    @test ids4.ghost = true

  end

  v = PVector(gids,map_parts(copy,gids),ids3;ids=:global)
  v = PVector(gids,map_parts(copy,gids),ids3;ids=:local)
  v = PVector(gids,map_parts(copy,gids),n;ids=:global)
  u = 2*v
  map_parts(u.values,v.values) do u,v
    @test u == 2*v
  end
  u = v + u
  map_parts(u.values,v.values) do u,v
    @test u == 3*v
  end

  w = similar(v)
  w = similar(v,Float64)
  w = similar(v,Float64,ids3)
  w = similar(typeof(v),ids3)
  w = zero(v)
  @test norm(w) == 0
  @test sum(w) == 0

  w = v .- u
  @test isa(w,PVector)
  w =  1 .+ v
  @test isa(w,PVector)
  w =  v .+ 1
  @test isa(w,PVector)
  w =  v .+ w .- u
  @test isa(w,PVector)
  w =  v .+ 1 .- u
  @test isa(w,PVector)

  w .= v .- u
  w .= v .- 1 .- u
  w .= u
  map_parts(w.values,u.values) do w,u
    @test w == u
  end

  u = PVector(1.0,ids2)
  w = PVector(3.0,ids3)

  @test oids_are_equal(u.rows,u.rows)
  @test hids_are_equal(u.rows,u.rows)
  @test lids_are_equal(u.rows,u.rows)
  @test oids_are_equal(u.rows,w.rows)
  @test !hids_are_equal(u.rows,w.rows)
  @test !lids_are_equal(u.rows,w.rows)

  w = v .- u
  @test isa(w,PVector)
  w =  v .+ w .- u
  @test isa(w,PVector)
  w =  v .+ 1 .- u
  @test isa(w,PVector)

  w .= v .- u
  w .= v .- 1 .- u
  w .= u
  map_parts(w.owned_values,u.owned_values) do w,u
    @test w == u
  end

  map_parts(parts,local_view(v,v.rows)) do part,v
    if part == 3
      v[1] = 6
    end
  end

  map_parts(parts,local_view(v,v.rows)) do part,v
    if part == 3
      @test v[1] == 6
    end
  end

  map_parts(parts,global_view(v,v.rows)) do part,v
    if part == 4
      v[9] = 6
    end
  end

  map_parts(parts,global_view(v,v.rows)) do part,v
    if part == 4
      @test v[9] == 6
    end
  end

  v = PVector{Float64}(undef,ids)
  fill!(v,1.0)

  v = PVector{Float64}(undef,ids)
  map_parts(parts,v.values,v.rows.partition) do part, values, partition
    for lid in 1:length(partition.lid_to_part)
      owner = partition.lid_to_part[lid]
      if owner == part
        values[lid] = 10*part
      end
    end
  end
  exchange!(v)
  map_parts(parts,v.values,v.rows.partition) do part, values, partition
    for lid in 1:length(partition.lid_to_part)
      owner = partition.lid_to_part[lid]
      @test values[lid] == 10*owner
    end
  end

  assemble!(v)

  cols = ids
  rows = cols

  values = map_parts(rows.partition,cols.partition) do rows, cols
    i = collect(1:num_lids(rows))
    j = i
    v = fill(2.0,length(i))
    sparse(i,j,v,num_lids(rows),num_lids(cols))
  end

  x = PVector{Float64}(undef,cols)
  fill!(x,3)
  b = PVector{Float64}(undef,rows)


  A = PSparseMatrix(values,rows,cols)
  A = PSparseMatrix(values,rows,cols,A.exchanger)
  mul!(b,A,x)

  map_parts(b.owned_values) do values
    @test all( values .== 6 )
  end

  exchange!(b)
  map_parts(b.values) do values
    @test all( values .== 6 )
  end

  exchange!(A)
  assemble!(A)

  I,J,V = map_parts(parts) do part
    if part == 1
      [1,2,1,2], [2,6,1,2], [1.0,2.0,30.0,10.0]
    elseif part == 2
      [3,3,4], [3,8,4], [10.0,2.0,30.0]
    elseif part == 3
      [5,5,6,7], [5,6,6,7], [10.0,2.0,30.0,1.0]
    else
      [9,9,8,10], [9,2,8,10], [10.0,2.0,30.0,50.0]
    end
  end
  A = PSparseMatrix(I,J,V,n,n;ids=:global)
  local_view(A,A.rows,A.cols)
  global_view(A,A.rows,A.cols)

  x = PVector{Float64}(undef,A.cols)
  fill!(x,1.0)
  y = A*x
  dy = y - y

  y = PVector(1.0,A.rows)
  x = IterativeSolvers.cg(A,y)

  x = PVector(0.0,A.cols)
  IterativeSolvers.cg!(x,A,y)
  fill!(x,0.0)

end

