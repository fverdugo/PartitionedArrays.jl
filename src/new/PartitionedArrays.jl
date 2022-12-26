using SparseArrays
using SparseMatricesCSR
using LinearAlgebra
using Printf
using CircularArrays
import MPI
import IterativeSolvers
import Distances

export prefix_sum!
export right_shift!
export jagged_array
export GenericJaggedArray
export JaggedArray
include("jagged_array.jl")

export nziterator
include("sparse_utils.jl")

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
export scan!
export reduction
export reduction!
export ExchangeGraph
export exchange
export exchange!
export allocate_exchange
include("primitives.jl")

export SequentialData
export with_sequential_data
include("sequential_data.jl")

export MPIData
export mpi_data
export with_mpi_data
include("mpi_data.jl")

export local_range
export PRange
export ConstantBlockSize
export VariableBlockSize
export get_local_to_global
export get_own_to_global
export get_ghost_to_global
export get_local_to_owner
export get_own_to_owner
export get_ghost_to_owner
export get_global_to_local
export get_global_to_own
export get_global_to_ghost
export get_own_to_local
export get_ghost_to_local
export get_local_to_own
export get_local_to_ghost
include("p_range.jl")

