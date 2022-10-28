
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

   a = map_one(+,b,rank;source=2)

   map(a,b,rank) do a,b,rank
       if rank == 2
           @assert a == b+rank
       else
           @assert a === nothing
       end
   end

   snd = b
   rcv = gather(snd;destination=2)
   display(rcv)

end
