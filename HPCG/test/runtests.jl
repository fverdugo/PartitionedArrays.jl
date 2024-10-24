module HPCGRunTests

using Test

@testset "hpcg_debug" begin
    include("hpcg_debug/runtests.jl")
end
@testset "hpcg_mpi" begin
    include("hpcg_mpi/runtests.jl")
end

end
