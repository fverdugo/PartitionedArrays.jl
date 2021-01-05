module RunTests

using Test

@testset "Sequential" begin include("sequential/runtests.jl") end

@testset "MPI" begin include("mpi/runtests.jl") end

end # module

