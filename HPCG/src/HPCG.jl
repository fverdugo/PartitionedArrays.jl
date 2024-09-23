module HPCG

using PartitionedArrays
using PartitionedSolvers
using LinearAlgebra
using DelimitedFiles
using Dates
using Statistics
using Primes
using DataStructures
using JSON
using SparseArrays
using SparseMatricesCSR
using Polyester
using LoopVectorization
import Base: iterate
using ThreadPinning

export hpcg_benchmark_mpi
export hpcg_benchmark_debug
export hpcg_benchmark
include("hpcg_benchmark.jl")

export build_matrix
export build_p_matrix
export ref_cg!
export pc_setup
export pc_solve!
include("hpcg_utils.jl")
include("compute_optimal_xyz.jl")
include("sparse_matrix.jl")

end # module HPCG
