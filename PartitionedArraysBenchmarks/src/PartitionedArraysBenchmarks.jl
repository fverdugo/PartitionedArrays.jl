module PartitionedArraysBenchmarks

using MPI
using PartitionedArrays
using PartitionedArrays: FakeTask, @fake_async
using PartitionedArrays: laplace_matrix, local_permutation
using PetscCall
using LinearAlgebra
using FileIO
using JLD2
using Mustache

include("helpers.jl")
include("benchmarks.jl")

end # module
