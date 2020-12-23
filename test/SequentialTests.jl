module SequentialTests

using DistributedDataDraft
using Test
using Gridap.Arrays: Table
using SparseArrays: sparse
using LinearAlgebra: mul!

nparts = 4
SequentialCommunicator(nparts) do comm

  @test num_parts(comm) == nparts
  @test num_workers(comm) == 1
  @test comm == comm
  @test i_am_master(comm)

  a = DistributedData{Int}(comm) do part
    10*part
  end

  b = DistributedData(a) do part, a
    20*part + a
  end

  @test a.parts == 10*collect(1:nparts)
  @test b.parts == 30*collect(1:nparts)

  do_on_parts(a,b) do part, a, b
    @test a == 10*part
    @test b == 30*part
  end

  @test gather(b) == 30*collect(1:nparts)

  c = bcast(comm,2)
  @test c.parts == fill(2,nparts)

  a2, b2 = DistributedData(comm) do part
    10*part, 30*part 
  end

  do_on_parts(a2,b2) do part, a2, b2
    @test a2 == 10*part
    @test b2 == 30*part
  end

  parts_rcv = DistributedData(comm) do part
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

  parts_snd = DistributedData(comm) do part
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

  data_snd = DistributedData(parts_snd) do part, parts_snd
    10*parts_snd
  end

  data_rcv = exchange(data_snd,parts_rcv,parts_snd)

  do_on_parts(data_rcv) do part, data_rcv
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

  data_snd = DistributedData(parts_snd) do part, parts_snd
    Table([ Int[i,part] for i in parts_snd])
  end

  data_rcv = exchange(data_snd,parts_rcv,parts_snd)

  do_on_parts(data_rcv) do part, data_rcv
    if part == 1
      r = [[1,2],[1,3]]
    elseif part == 2
      r = [[2,4]]
    elseif part == 3
      r = [[3,1],[3,2]]
    else
      r= [[4,1],[4,3]]
    end
    @test Table(r) == data_rcv
  end

  parts_snd_2 = discover_parts_snd(parts_rcv)
  do_on_parts(parts_snd,parts_snd_2) do part, parts_snd, parts_snd_2
    @test parts_snd == parts_snd_2
  end

  n = 10

  oids = DistributedRange(comm,n)

  do_on_parts(oids) do part, oids
    @test oids.part == part
  end

  lids = DistributedData(comm) do part
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

  do_on_parts(exchanger.parts_snd,exchanger.lids_snd) do part, parts_snd, lids_snd
    if part == 1
      parts = [2,4]
      ids = [[2],[1,3]]
    elseif part == 2
      parts = [1,3]
      ids = [[3],[3,2]]
    elseif part == 3
      parts = [1,4]
      ids = [[2,3],[2]]
    else
      parts = [2,3]
      ids = [[5],[5]]
    end
    @test parts == parts_snd
    @test ids == lids_snd
  end

  values = DistributedData(lids) do part, lids
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

  do_on_parts(values,lids) do part, values, lids
    for lid in 1:length(lids.lid_to_part)
      owner = lids.lid_to_part[lid]
      @test values[lid] == 10*owner
    end
  end

  exchanger_rcv = exchanger # receives data at ghost ids from remote parts
  exchanger_snd = reverse(exchanger_rcv) # sends data at ghost ids to remote parts

  values = DistributedData(lids) do part, lids
    values = fill(0.0,num_lids(lids))
    for lid in 1:length(lids.lid_to_part)
      values[lid] = 10*part
    end
    values
  end
  exchange!(values,exchanger_snd;reduce_op=+)
  exchange!(values,exchanger_rcv)

  ids = DistributedRange(n,lids)

  do_on_parts(ids) do part, ids
    @test ids.ngids == n
  end
  @test get_comm(ids) === comm
  @test num_parts(ids) == nparts
  @test num_gids(ids) == n

  v = DistributedVector{Float64}(undef,ids)
  fill!(v,1.0)

  v = DistributedVector{Float64}(undef,ids)
  do_on_parts(v.values,v.ids) do part, values, ids
    for lid in 1:length(ids.lid_to_part)
      owner = ids.lid_to_part[lid]
      if owner == part
        values[lid] = 10*part
      end
    end
  end
  exchange!(v)
  do_on_parts(v.values,v.ids) do part, values, ids
    for lid in 1:length(ids.lid_to_part)
      owner = ids.lid_to_part[lid]
      @test values[lid] == 10*owner
    end
  end

  assemble!(v)


  # Incremental creation of a DistributedVector

  # We start by allocating a seed, which 
  # only has data for owned ids corresponding
  # to a uniform partition of a range on n ids
  # TODO allow a user defined number of ids per part
  v = DistributedVectorSeed{Float64}(comm,n)

  # Set the data we want randomly
  do_on_parts(v) do part, v
    if part == 1
      setgid!(v,2.0,2)
      setgid!(v,10.0,10)
    elseif part == 2
      setgid!(v,1.0,1)
      setgid!(v,1.0,7)
    elseif part == 3
      setgid!(v,1.0,1)
      setgid!(v,1.0,4)
      setgid!(v,1.0,4)
    else
    end
  end

  # Build the vector
  # TODO Do this async
  v = DistributedVector(v)

  #u = v[ids]
  #@test u.ids === ids

  col_ids = ids
  row_ids = remove_ghost(col_ids)
  do_on_parts(row_ids) do part, row_ids
    @test all(i->i==part,row_ids.lid_to_part)
  end

  owned_values = DistributedData(row_ids,col_ids) do part, row_ids, col_ids
    i = collect(1:num_oids(row_ids))
    j = i
    v = fill(1.0,length(i))
    sparse(i,j,v,num_oids(row_ids),num_oids(col_ids))
  end

  ghost_values = DistributedData(row_ids,col_ids) do part, row_ids, col_ids
    sparse(Int[],Int[],Float64[],num_oids(row_ids),num_hids(col_ids))
  end

  x = DistributedVector{Float64}(undef,col_ids)
  fill!(x,3)
  b = DistributedVector{Float64}(undef,row_ids)

  A = DistributedSparseMatrix(owned_values,ghost_values,row_ids,col_ids)
  mul!(b,A,x)

  do_on_parts(b.values) do part, values
    @test all(  values .== 3 )
  end

  row_ids = col_ids
  b = DistributedVector{Float64}(undef,row_ids)
  A = DistributedSparseMatrix(owned_values,ghost_values,row_ids,col_ids)
  mul!(b,A,x)
  exchange!(b)

  do_on_parts(b.values) do part, values
    @test all(  values .== 3 )
  end

  #y = x[row_ids]
  #@test y.ids === row_ids

  #B = A[row_ids,col_ids]
  #@test B.col_ids === col_ids

  #B = A[row_ids,row_ids]
  #@test B.col_ids === row_ids

  #C = B[row_ids,col_ids]
  #@test C.col_ids === col_ids

  #mul!(b,B,y)

  P = AdditiveSchwarz(A)
  x = DistributedVector{Float64}(undef,col_ids)
  mul!(x,P,b)
  exchange!(x)

  do_on_parts(x.values) do part, values
    @test all(  values .== 3 )
  end

end # comm

end # Module
