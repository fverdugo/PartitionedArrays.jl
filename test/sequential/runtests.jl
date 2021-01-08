module SequentialTests

using Test

@testset "SequentialBackend" begin include("SequentialBackendTests.jl") end

@testset "Interfaces" begin include("InterfacesTests.jl") end

@testset "FDM" begin include("FDMTests.jl") end

@testset "FEMSA" begin include("FEMSATests.jl") end

end # module
