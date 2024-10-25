module PartitionedSolversTests

using PartitionedArrays
using PartitionedSolvers
using Test

@testset "PartitionedSolvers" begin
    @testset "interfaces" begin include("new/interfaces_tests.jl") end
    @testset "wrappers" begin include("new/wrappers_tests.jl") end
    @testset "smoothers" begin include("new/smoothers_tests.jl") end
    @testset "amg" begin include("new/amg_tests.jl") end
end

end # module
