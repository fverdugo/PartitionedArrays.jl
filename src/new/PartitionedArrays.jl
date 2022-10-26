using SparseArrays
using SparseMatricesCSR
using LinearAlgebra
using Printf
import MPI
import IterativeSolvers
import Distances


export with_backend
export linear_indices
export cartesian_indices
export unpack
export map_first
export gather!
export allocate_gather!
export gather
include("interfaces.jl")

export SequentialBackend
include("sequential_backend.jl")

