
using Test

function primitives_tests(distribute)

   rank = distribute(LinearIndices((2,2)))

   a_and_b = map(rank) do rank
       (2*rank,10*rank)
   end

   a, b = unpack(a_and_b)

   map(a,b,rank) do a,b,rank
       @test a == 2*rank
       @test b == 10*rank
   end

   a = map_one(+,b,rank;index=2)

   map(a,b,rank) do a,b,rank
       if rank == 2
           @assert a == b+rank
       else
           @assert a === nothing
       end
   end

   a = map_one(+,b,rank;index=:all)

   snd = b
   rcv = gather(snd;destination=2)
   map_one(rcv;index=2) do rcv
     @test rcv == [10,20,30,40]
   end

   snd2 = scatter(rcv;source=2)
   map(snd,snd2) do snd,snd2
       @test snd == snd2
   end
   @test typeof(snd) == typeof(snd2)

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

   rcv = emit(rank,source=2)
   map(rcv) do rcv
       @test rcv == 2
   end

   rcv = emit(snd,source=2)
   map(rcv) do rcv
     @test rcv == [1,2]
   end
   @test typeof(snd) == typeof(rcv)

   a = map(rank) do rank
       3*mod(rank,3)
   end
   b = scan(+,a,type=:inclusive,init=0)
   c = gather(b)
   map_one(c) do c
       @test c == [3,9,9,12]
   end

   b = scan(+,a,type=:exclusive,init=1)
   c = gather(b)
   map_one(c) do c
       @test c == [1,4,10,10]
   end

   b = copy(a)
   scan!(+,b,b,type=:inclusive,init=0)
   c = gather(b)
   map_one(c) do c
       @test c == [3,9,9,12]
   end

   b = copy(a)
   scan!(+,b,b,type=:exclusive,init=1)
   c = gather(b)
   map_one(c) do c
       @test c == [1,4,10,10]
   end

   r = reduction(+,rank,init=0)
   map_one(r) do r
       @test r == 10
   end
   r = reduction(+,rank,init=10,destination=:all)
   map(r) do r
       @test r == 20
   end

   r = copy(rank)
   reduction!(+,r,r,init=0,destination=2)
   map_one(r,index=2) do r
       @test r == 10
   end

   r = copy(rank)
   reduction!(+,r,r,init=10,destination=:all)
   map(r) do r
       @test r == 20
   end

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
   rcv = exchange(snd,graph)

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

end
