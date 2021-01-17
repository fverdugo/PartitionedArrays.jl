module PartitionedArrays

using SparseArrays
using LinearAlgebra
import MPI
import IterativeSolvers

export Backend
export distributed_run
export PData
export num_parts
export i_am_main
export map_parts
export get_part_ids
export get_backend
export MAIN
export get_part
export get_main_part
export gather!
export gather_all!
export gather
export gather_all
export scatter
export bcast
export reduce_main
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
export get_lid_to_gid
export get_lid_to_part
export get_gid_to_part
export get_oid_to_lid
export get_hid_to_lid
export get_lid_to_ohid
export get_gid_to_lid
export Exchanger
export allocate_rcv_buffer
export allocate_snd_buffer
export PRange
export add_gid
export add_gid!
export to_lid!
export to_gid!
export PVector
export local_view
export global_view
export async_assemble!
export assemble!
export PSparseMatrix
export nzindex
export nziterator
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
