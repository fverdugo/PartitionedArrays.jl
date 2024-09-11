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
import Base: iterate

export hpcg_benchmark_mpi
export hpcg_benchmark_debug
export hpcg_benchmark

export build_matrix
export build_p_matrix
export ref_cg!
export pc_setup
export pc_solve!
include("hpcg_benchmark.jl")

end # module HPCG
