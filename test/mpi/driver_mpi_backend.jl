using PartitionedArrays
using Test
using MPI

function main(parts)

  display(parts)

  nparts = num_parts(parts)
  @assert nparts == 4

  @test MPI.COMM_WORLD !== parts.comm
  _parts = get_part_ids(parts)
  @test _parts.comm === parts.comm

  #s = size(parts)
  #display(map_parts(part->s,parts))

  i_am_main(parts)

  values = map_parts(parts) do part
    10*part
  end

  @test size(values) == size(parts)

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

  @test size(data_snd) == size(parts)

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

  snd = map_parts(parts) do part
    if part == 1
      [1,2]
    elseif part == 2
      [2,3,4]
    elseif part == 3
      [5,6]
    else
      [7,8,9,10]
    end
  end
  rcv = gather(snd)
  map_parts(parts,rcv) do part, rcv
    if part == MAIN
      @test rcv == [[1,2],[2,3,4],[5,6],[7,8,9,10]]
    else
      @test rcv == Vector{Int}[]
    end
    @test isa(rcv,Table)
  end

  rcv = gather_all(snd)
  map_parts(rcv) do rcv
    @test rcv == [[1,2],[2,3,4],[5,6],[7,8,9,10]]
    @test isa(rcv,Table)
  end

  rcv = gather(parts)
  @test size(rcv) == size(parts)

  map_parts(parts,rcv) do part, rcv
    if part == MAIN
      @test rcv == collect(1:nparts)
    else
      @test length(rcv) == 0
    end
  end

  rcv = scatter(rcv)
  map_parts(parts,rcv) do part, rcv
    @test part == rcv
  end

  snd = map_parts(parts) do part
    if part == MAIN
      v = [[1,2],[2,3,4],[5,6],[7,8,9,10]]
    else
      v = Vector{Int}[]
    end
    Table(v)
  end
  rcv = scatter(snd)
  map_parts(parts,rcv) do part,rcv
    if part == 1
      r = [1,2]
    elseif part == 2
      r = [2,3,4]
    elseif part == 3
      r = [5,6]
    else
      r= [7,8,9,10]
    end
    @test r == rcv
  end

  snd = map_parts(parts) do part
    if part == MAIN
      v = [[1,2],[2,3,4],[5,6],[7,8,9,10]]
    else
      v = Vector{Int}[]
    end
    v
  end
  rcv = scatter(snd)
  map_parts(parts,rcv) do part,rcv
    if part == 1
      r = [1,2]
    elseif part == 2
      r = [2,3,4]
    elseif part == 3
      r = [5,6]
    else
      r= [7,8,9,10]
    end
    @test r == rcv
  end

  rcv = gather_all(parts)

  map_parts(rcv) do rcv
    @test rcv == collect(1:nparts)
  end

  @test get_part(rcv) == collect(1:nparts)

  snd = map_parts(parts) do part
    if part == MAIN
      [20,30,40]
    else
      Int[]
    end
  end

  rcv = emit(snd)
  map_parts(rcv) do rcv
    @test rcv == [20,30,40]
  end

  rcv = emit(parts)

  @test size(rcv) == size(parts)

  map_parts(rcv) do rcv
    @test rcv == 1
  end

  @test get_main_part(rcv) == MAIN

  @test get_part(parts,3) == 3

end

using MPI

MPI.Init()

nparts = 4
main(get_part_ids(MPIBackend(),nparts))

nparts = (2,2)
main(get_part_ids(MPIBackend(),nparts))

nparts = 2
_parts = get_part_ids(MPIBackend(),nparts)
if i_am_in(_parts)
  values = map_parts(_parts) do part
    10*part
  end
  @test size(values) == size(_parts)
  map_parts(_parts,values) do part, value
    @test 10*part == value
  end
end
