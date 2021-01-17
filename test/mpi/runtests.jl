module MPITests

using Test

@testset "Hello" begin include("HelloTests.jl") end

@testset "MPIBackend" begin include("MPIBackendTests.jl") end

@testset "Interfaces" begin include("InterfacesTests.jl") end

@testset "FDM" begin include("FDMTests.jl") end

@testset "FEMSA" begin include("FEMSATests.jl") end

end
