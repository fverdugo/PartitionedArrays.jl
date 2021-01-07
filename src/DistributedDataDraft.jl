module DistributedDataDraft

using SparseArrays
using LinearAlgebra
import MPI
import IterativeSolvers

export Backend
export distributed_run
export DistributedData
export num_parts
export i_am_master
export map_parts
export get_part_ids
export get_backend
export MASTER
export get_part
export get_master_part
export gather!
export gather_all!
export gather
export gather_all
export scatter
export bcast
export reduce_master
export reduce_all
export async_exchange!
export exchange!
export exchange
export async_exchange
export Table
export discover_parts_snd
export IndexSet
export num_gids
export num_lids
export num_oids
export num_hids
export oids_are_equal
export hids_are_equal
export lids_are_equal
export Exchanger
export allocate_rcv_buffer
export allocate_snd_buffer
export DistributedRange
export add_gid
export add_gid!
export to_lid!
export to_gid!
export DistributedVector
export local_view
export global_view
export async_assemble!
export assemble!
export DistributedSparseMatrix
export Jacobi

export SequentialBackend
export sequential

export MPIBackend
export mpi

include("Helpers.jl")

include("Interfaces.jl")

include("SequentialBackend.jl")

include("MPIBackend.jl")

end
