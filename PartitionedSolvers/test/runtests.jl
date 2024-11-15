module PartitionedSolversTests

using PartitionedArrays
using PartitionedSolvers
using Test

@testset "PartitionedSolvers" begin
    @testset "interfaces" begin include("interfaces_tests.jl") end
    @testset "wrappers" begin include("wrappers_tests.jl") end
    @testset "nonlinear_solvers" begin include("nonlinear_solvers_tests.jl") end
    @testset "ode_solvers" begin include("ode_solvers_tests.jl") end
    @testset "smoothers" begin include("smoothers_tests.jl") end
    @testset "amg" begin include("amg_tests.jl") end
end

end # module
