module HPCGDebugRunTests

using Test
using HPCG

@testset "hpcg_debug" begin
    include("hpcg_benchmark_tests.jl")
end

end #module
