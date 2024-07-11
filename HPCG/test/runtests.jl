module HPCGRunTests

using HPCG
using Test

@testset "hpcg_benchmark" begin
	include("hpcg_benchmark_tests.jl")
end

end
