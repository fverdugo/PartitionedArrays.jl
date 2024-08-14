module MPIArrayRunTests

using Test
using PartitionedArrays

@testset "mpi_array" begin include("mpi_array_tests.jl") end
@testset "primitives" begin include("primitives_tests.jl")  end
@testset "p_range_tests" begin include("p_range_tests.jl")  end
@testset "p_vector_tests" begin include("p_vector_tests.jl")  end
@testset "p_sparse_matrix_tests" begin include("p_sparse_matrix_tests.jl")  end
@testset "gallery" begin include("gallery_tests.jl")  end
@testset "block_arrays" begin include("block_arrays_tests.jl")  end
@testset "p_timer_tests" begin include("p_timer_tests.jl")  end
@testset "fdm_example" begin include("fdm_example.jl")  end
@testset "fem_example" begin include("fem_example.jl")  end

end #module
