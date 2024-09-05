
using Test

struct NonIsBitsType{T}
    data::Vector{T}
end
Base.:(==)(a::NonIsBitsType,b::NonIsBitsType) = a.data == b.data

function primitives_tests(distribute)

   rank = distribute(LinearIndices((2,2)))

   a_and_b = map(rank) do rank
       (2*rank,10*rank)
   end

   a, b = tuple_of_arrays(a_and_b)
   a_and_b_2 = array_of_tuples((a,b))
   map(a_and_b,a_and_b) do v1,v2
       @test v1 == v2
   end

   map(a,b,rank) do a,b,rank
       @test a == 2*rank
       @test b == 10*rank
   end

   a = map_main(+,b,rank;main=2)

   map(a,b,rank) do a,b,rank
       if rank == 2
           @assert a == b+rank
       else
           @assert a === nothing
       end
   end

   a = map_main(+,b,rank;main=:all)

   snd = b
   rcv = gather(snd;destination=2)
   map_main(rcv;main=2) do rcv
     @test rcv == [10,20,30,40]
   end
   gather!(rcv,snd;destination=2)
   map_main(rcv;main=2) do rcv
     @test rcv == [10,20,30,40]
   end

   snd2 = scatter(rcv;source=2)
   map(snd,snd2) do snd,snd2
       @test snd == snd2
   end
   @test typeof(snd) == typeof(snd2)
   #scatter!(snd2,rcv;source=2)

   snd = b
   rcv = gather(snd;destination=:all)
   map(rcv) do rcv
     @test rcv == [10,20,30,40]
   end

   snd = map(rank) do rank
     collect(1:rank)
   end
   rcv = gather(snd)
   snd2 = scatter(rcv)
   map(snd,snd2) do snd,snd2
       @test snd == snd2
   end
   @test typeof(snd) == typeof(snd2)

   rcv = gather(snd,destination=:all)
   map(rcv) do rcv
       @test rcv == [[1],[1,2],[1,2,3],[1,2,3,4]]
   end

   snd2 = map(rank) do rank
       NonIsBitsType([2])
   end
   rcv2 = gather(snd2)
   snd3 = scatter(rcv2)
   map(snd2,snd3) do snd2,snd3
       @test snd2 == snd3
   end

   #np = length(rank)
   #rcv3 = map_main(rank) do rank
   #    fill(NonIsBitsType([2]),np)
   #end
   #snd3 = allocate_scatter(rcv3)
   #scatter!(snd3,rcv3)
   #snd3 = scatter(rcv3)
   #rcv4 = gather(snd3)
   #map(rcv4,rcv2) do rcv4,rcv2
   #    @test rcv4 == rcv2
   #end

   rcv = multicast(rank,source=2)
   map(rcv) do rcv
       @test rcv == 2
   end

   rcv = multicast(snd,source=2)
   map(rcv) do rcv
     @test rcv == [1,2]
   end
   @test typeof(snd) == typeof(rcv)

   a = map(rank) do rank
       3*mod(rank,3)
   end
   b = scan(+,a,type=:inclusive,init=0)
   c = gather(b)
   map_main(c) do c
       @test c == [3,9,9,12]
   end

   b = scan(+,a,type=:exclusive,init=1)
   c = gather(b)
   map_main(c) do c
       @test c == [1,4,10,10]
   end

   #b = copy(a)
   #scan!(+,b,b,type=:inclusive,init=0)
   #c = gather(b)
   #map_main(c) do c
   #    @test c == [3,9,9,12]
   #end

   #b = copy(a)
   #scan!(+,b,b,type=:exclusive,init=1)
   #c = gather(b)
   #map_main(c) do c
   #    @test c == [1,4,10,10]
   #end

   r = reduction(+,rank,init=0)
   map_main(r) do r
       @test r == 10
   end
   r = reduction(+,rank,init=10,destination=:all)
   map(r) do r
       @test r == 20
   end
   @test reduce(+,rank) == 10
   @test reduce(+,rank,init=2) == 12
   @test sum(rank) == 10
   @test collect(rank) == [1 3; 2 4]

   #r = copy(rank)
   #reduction!(+,r,r,init=0,destination=2)
   #map_main(r,main=2) do r
   #    @test r == 10
   #end

   #r = copy(rank)
   #reduction!(+,r,r,init=10,destination=:all)
   #map(r) do r
   #    @test r == 20
   #end

   rcv_ids = map(rank) do rank
       if rank == 1
           [2,3]
       elseif rank == 2
           [4,]
       elseif rank == 3
           [1,2]
       else
           [1,3]
       end
   end

   snd_ids = map(rank) do rank
       if rank == 1
           [3,4]
       elseif rank == 2
           [1,3]
       elseif rank == 3
           [1,4]
       else
           [2]
       end
   end

   graph = ExchangeGraph(snd_ids,rcv_ids)

   snd = map(i->10*i,snd_ids)
   rcv = exchange(snd,graph) |> fetch

   map(rank,rcv) do rank, rcv
       if rank == 1
           r = [10,10]
       elseif rank == 2
           r = [20]
       elseif rank == 3
           r = [30,30]
       else
           r= [40,40]
       end
       @test r == rcv
   end

   graph2 = ExchangeGraph(snd_ids)
   map(==,graph2.rcv,graph.rcv)

   graph2 = ExchangeGraph(snd_ids,neighbors=graph)
   map(==,graph2.rcv,graph.rcv)

   graph2 = ExchangeGraph(snd_ids;find_rcv_ids=find_rcv_ids_gather_scatter)
   map(==,graph2.rcv,graph.rcv)

   if (isa(snd_ids,MPIArray))
    graph2 = ExchangeGraph(snd_ids;find_rcv_ids=find_rcv_ids_ibarrier)
    map(==,graph2.rcv,graph.rcv)
   end 

   snd = map(i->map(j->collect(1:j),i),snd_ids)
   rcv = exchange(snd,graph) |> fetch

   map(rank,rcv) do rank,rcv
       if rank == 1
           r = [[1],[1]]
       elseif rank == 2
           r = [[1,2]]
       elseif rank == 3
           r = [[1,2,3],[1,2,3]]
       else
           r= [[1,2,3,4],[1,2,3,4]]
       end
       @test r == rcv
   end

   parts = rank
   nparts = length(parts)
   @assert nparts == 4

   parts2 = linear_indices(parts)
   map(parts,parts2) do part1, part2
       @test part1 == part2
   end

   parts_rcv = map(parts) do part
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

   parts_snd = map(parts) do part
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

   data_snd = map(i->10*i,parts_snd)
   data_rcv = map(similar,parts_rcv)

   exchange!(
             data_rcv,
             data_snd,
             ExchangeGraph(parts_snd,parts_rcv)) |> fetch

   map(parts,data_rcv) do part, data_rcv
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

   a = reduction(+,parts,init=0)
   map_main(a) do a
       @test a == 1+2+3+4
   end

   b = reduction(+,parts,init=0,destination=:all)
   map(b) do b
       @test b == 1+2+3+4
   end
   @test reduce(+,parts,init=0) == 1+2+3+4
   @test sum(parts) == 1+2+3+4

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
   b = scan(+,a,init=0,type=:inclusive)
   map(parts,b) do part,b
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

   b = scan(+,a,init=1,type=:exclusive)
   map(parts,b) do part,b
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


   t = exchange(
                data_snd,
                ExchangeGraph(parts_snd, parts_rcv))

   data_rcv = fetch(t)
   map(parts,data_rcv) do part, data_rcv
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
                       ExchangeGraph(parts_snd, parts_rcv)) |> wait

   data_rcv = exchange(
                       data_snd,
                       ExchangeGraph(parts_snd, parts_rcv)) |> fetch

   map(parts,data_rcv) do part, data_rcv
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

   graph2 = ExchangeGraph(parts_rcv)
   parts_snd_2 = graph2.rcv
   map(parts_snd,parts_snd_2) do parts_snd, parts_snd_2
       @test parts_snd == parts_snd_2
   end
end
