module SequentialTests

using Test
using PartitionedArrays

@testset "backend" begin include("sequential_backend_tests.jl") end

@testset "interfaces" begin include("interfaces_tests.jl")  end

end #module
