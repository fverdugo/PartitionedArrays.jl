module SequentialTests

using DistributedDataDraft
using Test

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

  parts_snd_2 = discover_parts_snd(parts_rcv)
  do_on_parts(parts_snd,parts_snd_2) do part, parts_snd, parts_snd_2
    @test parts_snd == parts_snd_2
  end

  n = 10
  indices = DistributedIndexSet(comm,n) do part
    if part == 1
      IndexSet(n,1:5,fill(part,5))
    else
      IndexSet(n,6:10,fill(part,5))
    end
  end
  do_on_parts(indices) do part, indices
    @test indices.lid_to_owner == fill(part,5)
    @test indices.ngids == n
  end
  @test get_comm(indices) === comm
  @test num_parts(indices) == nparts
  @test num_gids(indices) == n

end 

end # Module
