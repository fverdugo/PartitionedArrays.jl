include("new/runtests.jl")

#=

module RunTests

using Test

@testset "IndexSets" begin include("IndexSetsTests.jl") end

@testset "SparseUtils" begin include("SparseUtilsTests.jl") end

@testset "Sequential" begin include("sequential/runtests.jl") end

@testset "MPI" begin include("mpi/runtests.jl") end

end # module

=#

