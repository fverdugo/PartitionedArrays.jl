using SparseArrays
using SparseMatricesCSR
using LinearAlgebra
using Printf
import MPI
import IterativeSolvers
import Distances

export prefix_sum!
export right_shift!
export jagged_array
export GenericJaggedArray
export JaggedArray
include("jagged_array.jl")

export linear_indices
export cartesian_indices
export unpack
export map_one
export gather
export gather!
export allocate_gather
export scatter
export scatter!
export allocate_scatter
export emit
export emit!
export allocate_emit
export scan
export reduction
include("primitives.jl")

export SequentialArray
include("sequential_array.jl")

