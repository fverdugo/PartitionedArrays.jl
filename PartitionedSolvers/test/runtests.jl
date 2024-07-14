module PartitionedSolversTests

using PartitionedArrays
using PartitionedSolvers
using Test

@testset "PartitionedSolvers" begin
    @testset "gallery" begin include("gallery_tests.jl") end
    @testset "smoothers" begin include("smoothers_tests.jl") end
    @testset "amg" begin include("amg_tests.jl") end
end

end # module
