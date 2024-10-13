module PartitionedSolversTests

using PartitionedArrays
using PartitionedSolvers
using Test

@testset "PartitionedSolvers" begin
    #@testset "smoothers" begin include("smoothers_tests.jl") end
    #@testset "amg" begin include("amg_tests.jl") end
    #@testset "nonlinear_solvers" begin include("nonlinear_solvers_tests.jl") end
    @testset "linear_solvers" begin include("linear_solvers_tests.jl") end
end

end # module
