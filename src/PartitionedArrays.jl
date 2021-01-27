module PartitionedArrays

using SparseArrays
using LinearAlgebra
using Printf
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
export emit
export reduce_main
export reduce_all
export iscan
export iscan_main
export iscan_all
export xscan
export xscan_main
export xscan_all
export async_exchange!
export exchange!
export exchange
export async_exchange
export Table
export get_data
export get_ptrs
export length_to_ptrs!
export rewind_ptrs!
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
export WithGhost
export with_ghost
export NoGhost
export no_ghost
export add_gid
export add_gid!
export to_lid!
export to_gid!
export PCartesianIndices
export PVector
export local_view
export global_view
export async_assemble!
export assemble!
export PSparseMatrix
export nzindex
export nziterator
export Jacobi
export PTimer
export tic!
export toc!
export print_timer

export SequentialBackend
export sequential

export MPIBackend
export mpi

include("Helpers.jl")

include("Interfaces.jl")

include("PTimers.jl")

include("SequentialBackend.jl")

include("MPIBackend.jl")

end
