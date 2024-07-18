module HPCG

using PartitionedArrays
using PartitionedSolvers
using LinearAlgebra
using DelimitedFiles
using Dates
using Statistics
using Primes
using DataStructures
import Base: iterate

export hpcg_benchmark_mpi
export hpcg_benchmark_debug
export hpcg_benchmark

export build_matrix
export build_p_matrix
export ref_cg!
export pc_setup
include("hpcg_benchmark.jl")

end # module HPCG
