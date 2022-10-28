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

export unpack
export map_one
export gather!
export allocate_gather
export gather
include("primitives.jl")

export SequentialArray
include("sequential_array.jl")

