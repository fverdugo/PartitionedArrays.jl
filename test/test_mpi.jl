
using DistributedDataDraft
using Test

nparts = 4
distributed_run(mpi,nparts) do parts

  display(parts)

  values = map_parts(parts) do part
    10*part.id
  end
  
  map_parts(parts,values) do part, value
    @test 10*part.id == value
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
    Table([ Int[i,part.id] for i in parts_snd])
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

end
