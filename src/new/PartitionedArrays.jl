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
export boundary_owner
export PRange
export ConstantBlockSize
export VariableBlockSize
export AbstractLocalIndices
export OwnAndGhostIndices
export LocalIndices
export PermutedLocalIndices
export GhostIndices
export OwnIndices
export get_n_local
export get_n_global
export get_n_ghost
export get_n_own
export get_owner
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
export replace_ghost
export union_ghost
export find_owner
export Assembler
export vector_assembler
export assemble!
export assembly_buffer_snd
export assembly_buffer_rcv
include("p_range.jl")

export get_local_values
export get_own_values
export get_ghost_values
export allocate_local_values
export OwnAndGhostValues
export PVector
export pvector
export pfill
export pzeros
export pones
export prand
export prandn
include("p_vector.jl")

