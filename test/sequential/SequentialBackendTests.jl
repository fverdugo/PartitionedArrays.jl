module SequentialBackendTests

using DistributedDataDraft
using Test

nparts = 4

parts = get_part_ids(sequential,nparts)

values = map_parts(parts) do part
  10*part
end

map_parts(parts,values) do part, value
  @test 10*part == value
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

t = async_exchange!(
  data_rcv,
  data_snd,
  parts_rcv,
  parts_snd)

map_parts(i->isa(i,Task),t)
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

data_snd = map_parts(parts,parts_snd) do part, parts_snd
  Table([ Int[i,part] for i in parts_snd])
end

data_rcv = exchange(data_snd,parts_rcv,parts_snd)

map_parts(parts,data_rcv) do part, data_rcv
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

a_and_b = map_parts(parts) do part
  10*part, part+10
end

a,b = a_and_b
map_parts(a_and_b,a,b) do a_and_b, a, b
  a1,b1 = a_and_b
  @test a == a1
  @test b == b1
end

rcv = gather(parts) 

map_parts(parts,rcv) do part, rcv
  if part == MASTER
    @test rcv == collect(1:nparts)
  else
    @test length(rcv) == 0
  end
end

@test get_master_part(rcv) == collect(1:nparts)

rcv = scatter(rcv)
map_parts(parts,rcv) do part, rcv
  @test part == rcv
end

rcv = gather_all(parts) 

map_parts(rcv) do rcv
  @test rcv == collect(1:nparts)
end

@test get_part(rcv) == collect(1:nparts)

rcv = bcast(parts)

map_parts(rcv) do rcv
  @test rcv == 1
end

end # module
