module DistributedDataDraft

using Gridap.Arrays: Table, length_to_ptrs!, rewind_ptrs!, lazy_map, PosNegReindex, PosNegPartition, Reindex
using Gridap.Algebra: scale_entries!
using SparseArrays: AbstractSparseMatrix, findnz, sparse
using FillArrays
using LinearAlgebra

export Communicator
export num_parts
export num_workers
export do_on_parts
export map_on_parts
export i_am_master
export OrchestratedCommunicator
export Communicator
export DistributedData
export get_comm
export get_part_type
export gather!
export gather
export scatter
export bcast
export get_distributed_data
export assemble!
export async_exchange!
export exchange!
export exchange
export discover_parts_snd
export num_oids
export num_hids
export IndexSet
export setgid!
export num_lids
export Exchanger
export allocate_rcv_buffer
export allocate_snd_buffer
export DistributedRange
export num_gids
export remove_ghost
export DistributedVector
export DistributedVectorSeed
export DistributedSparseMatrix
export AdditiveSchwarz

export SequentialCommunicator

include("Helpers.jl")

include("Interfaces.jl")

include("Sequential.jl")

end
