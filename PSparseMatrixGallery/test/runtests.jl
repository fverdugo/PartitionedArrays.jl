module PSparseMatrixGallery

using Test

@testset "debug_array" begin include("debug_array/runtests.jl") end
#@testset "mpi_array" begin include("mpi_array/runtests.jl") end

end # module
