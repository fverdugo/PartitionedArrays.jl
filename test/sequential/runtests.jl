module SequentialTests

using Test

@testset "SequentialBackend" begin include("SequentialBackendTests.jl") end

@testset "Interfaces" begin include("InterfacesTests.jl") end

end # module
