module RunTests

using Test

@testset "PartitionedArraysBenchmarks" begin
    @testset "benchmark_tests" begin include("benchmark_tests.jl") end
    @testset "helpers_tests" begin include("helpers_tests.jl") end
    @testset "analysis" begin include("analysis.jl") end
end

end # module
