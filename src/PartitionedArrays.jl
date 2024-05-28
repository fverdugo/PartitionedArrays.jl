module PartitionedArrays

using SparseArrays
using SparseMatricesCSR
using LinearAlgebra
using Printf
using CircularArrays
import MPI
import IterativeSolvers
import Distances

export length_to_ptrs!
export rewind_ptrs!
export jagged_array
export GenericJaggedArray
export JaggedArray
include("jagged_array.jl")

export nziterator
export nzindex
export compresscoo
export indextype
export sparse_matrix
export sparse_matrix!
include("sparse_utils.jl")

export linear_indices
export cartesian_indices
export tuple_of_arrays
export i_am_main
export MAIN
export map_main
export gather
export gather!
export allocate_gather
export scatter
export scatter!
export allocate_scatter
export multicast
export multicast!
export allocate_multicast
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
export find_rcv_ids_gather_scatter
include("primitives.jl")

export DebugArray
export with_debug
include("debug_array.jl")

export MPIArray
export distribute_with_mpi
export with_mpi
export find_rcv_ids_ibarrier
include("mpi_array.jl")

export PRange
export uniform_partition
export variable_partition
export partition_from_color
export trivial_partition
export renumber_partition
export AbstractLocalIndices
export OwnAndGhostIndices
export LocalIndices
export permute_indices
export PermutedLocalIndices
export GhostIndices
export OwnIndices
export local_length
export global_length
export ghost_length
export own_length
export part_id
export local_to_global
export own_to_global
export ghost_to_global
export local_to_owner
export own_to_owner
export ghost_to_owner
export global_to_local
export global_to_own
export global_to_ghost
export own_to_local
export ghost_to_local
export local_to_own
export local_to_ghost
export replace_ghost
export remove_ghost
export union_ghost
export find_owner
export assemble!
export to_local!
export to_global!
export partition
export assembly_graph
export assembly_neighbors
export assembly_local_indices
export map_local_to_global!
export map_global_to_local!
export map_ghost_to_global!
export map_global_to_ghost!
export map_own_to_global!
export map_global_to_own!
include("p_range.jl")

export local_values
export own_values
export ghost_values
export OwnAndGhostVectors
export PVector
export pvector
export old_pvector!
export pvector
export pvector!
export pfill
export pzeros
export pones
export prand
export prandn
export consistent!
export assemble
export consistent
export repartition
export repartition!
export find_local_indices
include("p_vector.jl")

export OldPSparseMatrix
export SplitMatrix
export PSparseMatrix
export old_psparse
export psparse
export psparse!
export split_format
export split_format!
export old_psparse!
export own_ghost_values
export ghost_own_values
export own_own_values
export ghost_ghost_values
export psystem
export psystem!
export dense_diag
export dense_diag!
export rap
export rap!
export spmm
export spmm!
export spmtm
export spmtm!
export centralize
include("p_sparse_matrix.jl")

export PTimer
export tic!
export toc!
export statistics
include("p_timer.jl")

end # module
