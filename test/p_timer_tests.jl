using PartitionedArrays
using Test
using LinearAlgebra
using Random
using Distances

function p_timer_tests(distribute)

    parts = distribute(LinearIndices((4,)))

    t = PTimer(parts)

    tic!(t;barrier=true)
    toc!(t,"Phase 1")
    toc!(t,"Phase 3")
    sleep(0.2)
    toc!(t,"Phase 143")
    tic!(t;barrier=true)
    sleep(0.4)
    toc!(t,"Matrix Assembly")

    map_main(t.data) do data
        open("times.txt","w") do io
            println(io,data)
        end
    end

    display(t)

    t = PTimer(parts,verbose=true)

    tic!(t;barrier=true)
    toc!(t,"Phase 1")
    toc!(t,"Phase 3")
    sleep(0.2)
    toc!(t,"Phase 143")
    tic!(t;barrier=true)
    sleep(0.4)
    toc!(t,"Matrix Assembly")

    display(t)

end
