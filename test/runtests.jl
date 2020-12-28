using DistributedDataDraft
using Test

@testset "Interfaces" begin include("InterfacesTests.jl") end

@testset "SequentialBackend" begin include("SequentialBackendTests.jl") end

@testset "MPIBackend" begin include("MPIBackendTests.jl") end

