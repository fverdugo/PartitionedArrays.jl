module SequentialTests

using Test
using PartitionedArrays

include(joinpath("..","interfaces_tests.jl"))

@testset "defaults" begin with_backend(interfaces_tests,nothing) end

backend = SequentialBackend()

@testset "backend" begin include("sequential_backend_tests.jl") end

@testset "interfaces" begin with_backend(interfaces_tests,backend) end

end #module
