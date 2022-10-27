using SparseArrays
using SparseMatricesCSR
using LinearAlgebra
using Printf
import MPI
import IterativeSolvers
import Distances

export exclusive_scan!
export inclusive_scan!
export rewind!
export jagged_array
export GenericJaggedArray
export JaggedArray
include("jagged_array.jl")

export with_backend
export linear_indices
export cartesian_indices
export unpack
export map_one
export gather!
export allocate_gather
export gather
include("interfaces.jl")

export SequentialBackend
include("sequential_backend.jl")

