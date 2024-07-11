module HPCG

using PartitionedArrays
using PartitionedSolvers
using LinearAlgebra
using Test
using SparseArrays
using IterativeSolvers
using BenchmarkTools
using DelimitedFiles

export build_pmatrix
export restrict_operator
export pc_setup
export pc_solve!
export restrict!
export prolongate!
export multigrid_preconditioner!
export HPCG_benchmark
include("hpcg_benchmark.jl")

export compute_optimal_shape_XYZ
include("compute_optimal_xyz.jl")

end # module HPCG
