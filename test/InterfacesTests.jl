module InterfacesTests

using DistributedDataDraft
using Test

nparts = 4
distributed_run(sequential,nparts) do parts

  parts2 = get_parts(sequential,nparts)
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

end

end # module
