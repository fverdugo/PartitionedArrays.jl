
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

   rcv = emit(snd,source=2)
   map(rcv) do rcv
     @test rcv == [1,2]
   end
   @test typeof(snd) == typeof(rcv)

   a = map(rank) do rank
       3*mod(rank,3)
   end
   b = inclusive_scan(+,a,destination=3)

   c = exclusive_scan(+,a,init=1,destination=:all)
   map(c) do c
       @test c == [1,4,10,10]
   end

end
